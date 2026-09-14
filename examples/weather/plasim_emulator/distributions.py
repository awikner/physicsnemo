# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hurdle (zero-inflated) Gamma distribution for PlaSim diagnostics.

Both diagnostics this emulator predicts are non-negative with a point mass at
zero, measured on sim52 year 50:

    pr_6h   ~31% exact zeros,  mean 2.4 mm/day,  p99 24.3,  max 89.4
    -evap   ~10% exact zeros,  mean 2.2 mm/day,  p99 10.2,  max 25.0

A Gaussian cannot represent either (it puts mass below zero and smears the
point mass), and a plain L2 loss on z-scored precipitation collapses toward the
conditional mean - which is exactly the over-smoothing that makes an emulator
useless for extremes.  The hurdle-Gamma factorises the problem instead:

    P[Y = 0]   = 1 - p
    P[Y > 0]   = p,  with  Y | Y>0 ~ Gamma(k, theta)

so the "does it rain" and "how hard" questions get separate parameters, and the
exceedance probability a rare-event pipeline needs is closed form.

The Gamma is fitted to values scaled by a per-channel constant (roughly the mean
of the positive part) so k and theta stay O(1); this is numerically far better
behaved than fitting a Gamma in log1p space, and keeps exceedance probabilities
exact under a simple rescaling of the threshold.
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-6
MAX_CONC = 1.0e4  # clamp on k, theta to keep lgamma finite under bf16 autocast


class HurdleGamma:
    """Wraps 3 raw network channels as a hurdle-Gamma per grid point.

    Parameters
    ----------
    raw : Tensor
        ``[..., 3, H, W]`` unconstrained network output: logit(p), then two
        pre-softplus channels for the Gamma concentration and scale.
    scale : float
        Physical scale the Gamma was fitted in; ``y_scaled = y / scale``.
    """

    def __init__(self, raw: torch.Tensor, scale: float = 1.0):
        if raw.shape[-3] != 3:
            raise ValueError(f"expected 3 param channels, got {raw.shape[-3]}")
        self.logit_p = raw[..., 0, :, :]
        # softplus keeps k, theta strictly positive; +EPS avoids a hard zero.
        self.k = torch.clamp(F.softplus(raw[..., 1, :, :]) + EPS, max=MAX_CONC)
        self.theta = torch.clamp(F.softplus(raw[..., 2, :, :]) + EPS, max=MAX_CONC)
        self.scale = scale

    @property
    def p(self) -> torch.Tensor:
        """P[Y > 0]."""
        return torch.sigmoid(self.logit_p)

    def nll(self, y: torch.Tensor) -> torch.Tensor:
        """Negative log likelihood, elementwise. `y` in physical units, >= 0."""
        ys = y / self.scale
        wet = ys > 0

        # Bernoulli part, computed in logit space for stability.
        #   wet  -> -log(p)      = softplus(-logit_p)
        #   dry  -> -log(1 - p)  = softplus( logit_p)
        bern = torch.where(
            wet, F.softplus(-self.logit_p), F.softplus(self.logit_p)
        )

        # Gamma part, only where wet. Clamp inside the log so the dry branch
        # never produces a NaN that would poison the gradient via `where`.
        ys_safe = torch.clamp(ys, min=EPS)
        log_gamma_pdf = (
            (self.k - 1.0) * torch.log(ys_safe)
            - ys_safe / self.theta
            - self.k * torch.log(self.theta)
            - torch.lgamma(self.k)
        )
        gam = torch.where(wet, -log_gamma_pdf, torch.zeros_like(log_gamma_pdf))
        return bern + gam

    def mean(self) -> torch.Tensor:
        """E[Y] = p * k * theta, in physical units."""
        return self.p * self.k * self.theta * self.scale

    def exceedance(self, threshold: float) -> torch.Tensor:
        """P[Y > threshold] in physical units, via the regularized upper
        incomplete gamma. Requires torch>=2.4 for `igammac`."""
        ts = torch.as_tensor(threshold / self.scale, device=self.k.device,
                             dtype=self.k.dtype)
        if ts <= 0:
            return self.p
        return self.p * torch.special.gammaincc(self.k, ts / self.theta)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw per grid point, in physical units.

        Uses the reparameterised Gamma sampler, then masks by a Bernoulli draw.

        CAVEAT: `torch._standard_gamma` takes no generator, so `generator` seeds
        only the Bernoulli part. For a fully reproducible draw, seed the global
        RNG (`torch.manual_seed`) as well. This does not affect AI-RES's
        determinism requirement, which is about the *initial-condition* noise
        seed - that path goes through `PlasimEmulator.draw_noise`, which is fully
        generator-controlled.
        """
        u = torch.rand(self.p.shape, device=self.p.device, dtype=self.p.dtype,
                       generator=generator)
        wet = (u < self.p).to(self.k.dtype)
        g = torch._standard_gamma(self.k) * self.theta
        return wet * g * self.scale

    def rsample(self, tau: float = 0.1, generator: torch.Generator | None = None
                ) -> torch.Tensor:
        """REPARAMETERISED draw - differentiable w.r.t. all three parameters.

        `sample()` is not usable inside a loss: its wet/dry gate is a hard
        Bernoulli comparison, which is discrete and passes no gradient to `p`.
        Here the gate is a Gumbel-sigmoid (Concrete) relaxation at temperature
        `tau`, and the Gamma draw uses `torch._standard_gamma`, whose backward
        w.r.t. concentration PyTorch implements. That makes a sampled objective
        such as the daily-aggregate CRPS trainable end to end.

        Low `tau` approaches the hard gate but sharpens gradients; 0.1 is a
        reasonable default.
        """
        u = torch.rand(self.p.shape, device=self.p.device, dtype=self.p.dtype,
                       generator=generator).clamp(1e-6, 1 - 1e-6)
        logistic = torch.log(u) - torch.log1p(-u)
        wet = torch.sigmoid((self.logit_p + logistic) / tau)
        g = torch._standard_gamma(self.k) * self.theta
        return wet * g * self.scale

    def crps_samples(self, y: torch.Tensor, n: int = 8,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample-based CRPS estimate (kernel form).

        CRPS = E|X - y| - 0.5 * E|X - X'|, estimated with `n` draws. More
        expensive than `nll` - the training loop defaults to NLL and keeps this
        for validation, where an unbiased proper score is worth the cost.
        """
        xs = torch.stack([self.sample(generator) for _ in range(n)], dim=0)
        term1 = (xs - y.unsqueeze(0)).abs().mean(dim=0)
        term2 = (xs.unsqueeze(0) - xs.unsqueeze(1)).abs().mean(dim=(0, 1))
        return term1 - 0.5 * term2


def fit_scale(values) -> float:
    """Scale constant for a diagnostic: the mean of its strictly positive part.

    Accepts a torch Tensor or a numpy array.
    """
    import numpy as np

    v = values if isinstance(values, np.ndarray) else values.detach().cpu().numpy()
    pos = v[v > 0]
    return float(pos.mean()) if pos.size else 1.0
