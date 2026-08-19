# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Spectrally shaped latent amplitude for Rolling Stochastic Interpolants.

ERDM's progressive schedule encodes "later = more uncertain" with a single
number per window slot. But error growth in chaotic fluids is spectrally
structured -- perturbations grow fastest at small scales and cascade upscale --
so a scalar white level simultaneously over-corrupts the large scales and
under-resolves where the conditional spread actually lives. The interpolant's
``Gamma`` is a free operator rather than a scalar, so it can be shaped:

    Gamma(tau, l) = gamma_0 * g(l) * h(tau, l)

turning the 1-D progressive schedule into a 2-D schedule over
(lead time, wavenumber).

**The envelope g(l)** is fit offline to the spherical power spectrum of the
model's own one-step residual (``tools/data/amip/make_rsi_spectrum.py``), then
normalized to unit band-mean so it REDISTRIBUTES amplitude across scales rather
than rescaling it -- the overall level stays ``gamma_0``'s job. A caution on
what it is: the increment spectrum is dominated by deterministic advection at
large scales, so it is not literally the conditional spread. It is still the
right default -- it is the scale on which the entering slot's anchor is wrong,
and it keeps the velocity regression well-conditioned band by band -- but a
narrower "true spread" envelope is a separate ablation, not a bug fix.

**The release profile h(tau, l)** controls when each band's uncertainty is
resolved as a slot crosses the window:

    h(tau, l) = h_1 ** (tau ** e_l),    e_l = 1 + sharpness * (l / lmax)

so ``h(0, l) = 1`` and ``h(1, l) = h_1`` for every band, while a larger ``e_l``
holds a band near its full amplitude until late in the traverse. With
``sharpness > 0`` that makes high-l bands resolve LATE (near the window front)
and low-l bands resolve early -- the physical prior that large scales are
pinned down almost immediately while small scales stay uncertain up to short
lead times.

The scalar case is an exact special case: ``g == 1`` and ``sharpness == 0`` give
``Gamma = gamma_0 * h_1**tau``, which with ``h_1 = gamma_1/gamma_0`` is exactly
the scheduler's geometric scalar profile. That equality is a unit test
(``test_rsi_spectral.py``), so turning spectral shaping on cannot silently
change the overall noise level.

Everything is diagonal in spectral space, so ``Gamma``, its inverse and its
tau-derivative are all one coefficient multiply between a forward and an
inverse transform.
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SphericalSpectralFilter(nn.Module):
    r"""Diagonal-in-spherical-harmonics latent amplitude ``Gamma(tau, l)``.

    Parameters
    ----------
    nlat, nlon
        The state grid. The transform is built for exactly this grid; a
        mismatch at call time raises rather than interpolating.
    gamma_0, gamma_1
        Endpoint amplitudes, as in the scalar profile. ``h_1 = gamma_1/gamma_0``.
    envelope
        ``g``: ``(L,)`` shared across channels, or ``(C, L)`` per channel. It is
        normalized to unit band-mean on load. ``None`` means ``g == 1``.
    sharpness
        How much later high-l bands are released than low-l ones. ``0``
        reproduces the scalar schedule exactly.
    lmax
        Bandwidth. Defaults to ``nlat // 2`` -- the exactness limit of
        equiangular quadrature, and the largest value for which the
        analysis-synthesis pair is a true projection (see ``__init__``).

    Notes
    -----
    The operator lives on the band-limited subspace, so ``apply`` acts as
    ``Gamma . P`` for the projector ``P = isht . sht``. That is not a
    limitation to work around: the latent is band-limited by construction, so
    ``E[z | x]`` lies in that subspace too, and projecting the network's
    ``zhat`` onto it is the correct thing to do. Identities like the scalar
    reduction and ``Gamma^{-1} Gamma = I`` therefore hold ON that subspace.
    """

    def __init__(self, nlat, nlon, *, gamma_0, gamma_1, envelope=None,
                 sharpness=2.0, lmax=None, eps=1e-12):
        super().__init__()
        from torch_harmonics import InverseRealSHT, RealSHT

        self.nlat, self.nlon = int(nlat), int(nlon)
        self.gamma_0 = float(gamma_0)
        self.gamma_1 = float(gamma_1)
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        if not 0.0 < self.gamma_1 < self.gamma_0:
            raise ValueError(
                f"need 0 < gamma_1 < gamma_0, got gamma_0={gamma_0}, "
                f"gamma_1={gamma_1}. Gamma(0) > 0 is mandatory (degenerate "
                f"couplings) and Gamma(1) > 0 is the emission noise floor; the "
                f"spectral filter also takes their ratio as a log."
            )
        self.h_1 = self.gamma_1 / self.gamma_0

        # lmax = nlat // 2, NOT nlat. Equiangular quadrature is only exact up
        # to half the latitude count: measured on 16/45/64-point grids, the
        # analysis-synthesis pair at lmax = nlat is aliased badly enough that
        # ||P^2 - P|| ~ 0.5 (of a field of scale ~3), while at nlat // 2 it is a
        # clean projection to ~3e-7. That matters here beyond accuracy —
        # Gamma and Gamma^{-1} are built from the same pair, so on an aliased
        # transform the score identity simply would not hold, silently.
        lmax = int(lmax) if lmax is not None else max(2, self.nlat // 2)
        self.sht = RealSHT(nlat=self.nlat, nlon=self.nlon, lmax=lmax,
                           grid="equiangular")
        self.isht = InverseRealSHT(nlat=self.nlat, nlon=self.nlon, lmax=lmax,
                                   grid="equiangular")
        # Ask the transform for its coefficient widths rather than assuming --
        # torch_harmonics changed the mmax convention between 0.8 and 0.9 (see
        # SphereNoiseGenerator's note).
        self.l_max = int(getattr(self.sht, "lmax", lmax))
        self.m_max = int(getattr(self.sht, "mmax", lmax))

        # Per-degree release exponent e_l, shape (L, 1) to broadcast over m.
        degree = torch.arange(self.l_max, dtype=torch.float32)
        e_l = 1.0 + self.sharpness * (degree / max(self.l_max - 1, 1))
        self.register_buffer("e_l", e_l[:, None])

        if envelope is None:
            g = torch.ones(self.l_max)
        else:
            g = torch.as_tensor(envelope, dtype=torch.float32)
            if g.shape[-1] != self.l_max:
                raise ValueError(
                    f"envelope has {g.shape[-1]} degrees but the transform has "
                    f"{self.l_max}; refit it for this grid rather than padding."
                )
            # Unit band-mean: g redistributes across scales, it does not
            # rescale. The overall level stays gamma_0's job, so an envelope
            # swap cannot silently move the noise magnitude.
            g = g / g.mean(dim=-1, keepdim=True).clamp(min=self.eps)
        # (L, 1) or (C, L, 1)
        self.register_buffer("g", g[..., None])

        logger.info(
            "SphericalSpectralFilter: grid=%sx%s lmax=%s mmax=%s gamma=[%s -> %s] "
            "sharpness=%s envelope=%s",
            self.nlat, self.nlon, self.l_max, self.m_max, self.gamma_0,
            self.gamma_1, self.sharpness,
            "unit" if envelope is None else tuple(g.shape[:-1]),
        )

    # ------------------------------------------------------------------
    def _coeff(self, tau_flat):
        """Amplitude coefficients for a flat batch of taus -> (n, [C,] L, 1)."""
        tau = tau_flat.clamp(0.0, 1.0).float()[:, None, None]        # (n, 1, 1)
        e = self.e_l.float()                                         # (L, 1)
        h = self.h_1 ** (tau ** e)                                   # (n, L, 1)
        g = self.g.float()
        if g.dim() == 3:                                             # per channel
            return self.gamma_0 * g[None] * h[:, None]               # (n, C, L, 1)
        return self.gamma_0 * g * h                                  # (n, L, 1)

    def _dcoeff(self, tau_flat):
        """d/dtau of :meth:`_coeff`, same shape."""
        tau = tau_flat.clamp(self.eps, 1.0).float()[:, None, None]
        e = self.e_l.float()
        h = self.h_1 ** (tau ** e)
        dh = h * math.log(self.h_1) * e * tau ** (e - 1.0)
        g = self.g.float()
        if g.dim() == 3:
            return self.gamma_0 * g[None] * dh[:, None]
        return self.gamma_0 * g * dh

    def _transform(self, v, coeff):
        """Apply a diagonal spectral operator to ``v`` (b, W, C, H, W).

        The round trip runs in fp32 with autocast **disabled**, not merely on an
        fp32 input. Casting the input is not sufficient: under an enclosing
        ``torch.autocast(bf16)`` the transform's own internal matmuls are
        downcast, and ``torch_harmonics`` then hits
        ``view_as_complex``, which has no bfloat16 kernel -- a hard
        RuntimeError, and only on GPU (found on a GH200, 2026-08-18). Disabling
        autocast for the region is also the right call numerically: a bf16 SHT
        round trip loses far more than the transform costs, and the inverse of
        a small ``gamma_1`` would quantize badly.
        """
        b, W, C, H, Wd = v.shape
        if (H, Wd) != (self.nlat, self.nlon):
            raise ValueError(
                f"spectral Gamma was built for a {self.nlat}x{self.nlon} grid "
                f"but got {H}x{Wd}. Rebuild the filter for this grid."
            )
        dev = v.device.type
        with torch.autocast(device_type=dev, enabled=False):
            spec = self.sht(v.float().reshape(b * W, C, H, Wd))      # (bW, C, L, M)
            if coeff.dim() == 3:
                coeff = coeff[:, None]                               # broadcast over C
            out = self.isht(spec * coeff.to(spec.real.dtype))
        return out.reshape(b, W, C, H, Wd).to(v.dtype)

    # ------------------------------------------------------------------
    def apply(self, v, tau):
        """Gamma(tau) v."""
        return self._transform(v, self._coeff(tau.reshape(-1)))

    def apply_dot(self, v, tau):
        """Gamma'(tau) v."""
        return self._transform(v, self._dcoeff(tau.reshape(-1)))

    def apply_delta(self, v, tau_a, tau_b):
        """(Gamma(tau_b) - Gamma(tau_a)) v -- the exact coefficient increment.

        One transform, not two: the operator is diagonal, so the increment is
        just the coefficient difference.
        """
        coeff = self._coeff(tau_b.reshape(-1)) - self._coeff(tau_a.reshape(-1))
        return self._transform(v, coeff)

    def apply_inv(self, v, tau):
        """Gamma(tau)^{-1} v, regularized."""
        return self._transform(
            v, 1.0 / self._coeff(tau.reshape(-1)).clamp(min=self.eps))

    def band_amplitude(self, tau):
        """Band-mean |Gamma(tau)| -- the scalar the spectral case reduces to."""
        return self._coeff(tau.reshape(-1)).mean(dim=(-2, -1)).reshape(tau.shape)
