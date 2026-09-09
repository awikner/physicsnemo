# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Real-model probes for the RSI long-rollout drift diagnosis.

Tests 1, 2 and 4 of ``docs/dev/context/rsi-drift-diagnosis-report.md``, run
against a trained RSI or ERDM forecaster with the eval suite's own dataset,
weights, window assembly and forcing alignment (mirrors
``validate_diffusion.DiffusionRolloutValidator._rollout_window`` exactly:
RSI's first window is rows ``t..t+W`` (anchor + W frames), ERDM's is
``t+1..t+W``; trajectory slot ``i`` is the forcing at row ``t+i``, so window
slot ``w`` of roll ``k`` sees the lag-1 forcing ``traj[k+w]``).

One process, one GPU. Select probes with ``+probes.which=[readout,cascade,flush]``:

``readout``
    Teacher-forced per-slot, per-channel readout statistics at the sampler's
    own head-evaluation times (global ``t = 0.5`` for the shipped 2-step Heun,
    where both the emitted frame and the fresh-slot anchor are read out; also
    ``t = 0``). For every slot and channel: ``std(y_hat)/std(y)`` of the
    spatial anomaly, the regression slope of ``y_hat`` on ``y``, the RMSE, the
    response of the readout's spatial mean to a uniform offset of the whole
    window (level gain), and the response of its anomaly amplitude to a
    uniform shrink of the window's anomaly pattern. ERDM: the same for the
    denoiser output ``D`` at the ``t = 0.5`` sigma staircase.
``cascade``
    Free rollout of ``probes.cascade_rolls`` frames from ``probes.ic_dates``.
    Per roll and channel: spatial-anomaly std and spatial mean of the emitted
    frame and of the truth at the same row (normalized units). RSI also writes
    the fresh-slot anchor and per-slot window statistics through the
    scheduler's ``RSI_TRACE_PATH`` hook.
``flush``
    Multi-roll restoring-force probe (replaces the brief's one-roll Test A):
    perturb the initial window (anomaly pattern x ``probes.flush_shrink``;
    uniform offset ``probes.flush_offset`` in normalized units), roll
    ``probes.flush_rolls`` frames with the SAME latent draws as an unperturbed
    reference, and report ``rms(perturbed - reference) / rms(perturbation)``
    per variable group, against the model's own two-seed decorrelation floor.
    Done from the true IC window and from the free-run window after
    ``probes.flush_k0`` rolls (the off-manifold case).

Example::

    python examples/weather/ai_rossby/rsi_drift_probes.py \
        --config-dir=examples/weather/ai_rossby/conf --config-name=config \
        run_name=probe_rsi model=amip_combined_rsi_sstpred loss=rsi \
        loss.window_size=6 dataset=amip_dailyavg_coarse_multiyear \
        validation=eval_suite seed=0 wandb.enabled=False \
        ++model.forecaster.checkpoint=$R/checkpoints_prod24_b40/RollingDiTWrapper.0.24.mdlus \
        +probes.which=[readout,cascade] +probes.use_ema=true \
        '+probes.ic_dates=[1996-01-01,1996-04-01,1996-07-01,1996-10-01]' \
        +probes.out=$R/probes_e24/rsi

Outputs ``<out>/<probe>.pt`` (all arrays) and ``<out>/summary.json``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger("rsi_drift_probes")

_TRAJ_KEYS = ("surface_in", "varying_boundary", "calendar", "constant_boundary")


# --------------------------------------------------------------------------- #
# Dataset windows (mirrors DiffusionRolloutValidator._stack_window / _stack)
# --------------------------------------------------------------------------- #
def _fetch(ds, t: int) -> dict:
    try:
        return ds[(int(t), 1)]
    except (TypeError, KeyError):
        return ds[int(t)]


def _stack_frames(ds, rows, keys=None) -> dict:
    frames = [_fetch(ds, r) for r in rows]
    out = {}
    for k, v0 in frames[0].items():
        if keys is not None and k not in keys:
            continue
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([f[k] for f in frames], dim=0)
        else:
            out[k] = v0
    return out


def _batch_windows(ds, ics, first_offset: int, n_frames: int, step: int, keys=None) -> dict:
    """(B, n, ...) dict; frame ``i`` of IC ``t`` is store row ``t + (first_offset + i) * step``."""
    per = [
        _stack_frames(ds, [t + (first_offset + i) * step for i in range(n_frames)], keys)
        for t in ics
    ]
    out = {}
    for k, v0 in per[0].items():
        out[k] = torch.stack([w[k] for w in per], dim=0) if isinstance(v0, torch.Tensor) else v0
    return out


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _forcing_traj(ds, wrapper, ics, traj_len: int, step: int, device):
    """``(c_grid_traj, c_scalar_traj)`` for rows ``t .. t + traj_len - 1``."""
    traj = _batch_windows(ds, ics, 0, traj_len, step, keys=_TRAJ_KEYS)
    c_grid = wrapper.pack_window_c_grid(
        {
            "surface_in": traj["surface_in"],
            "constant_boundary": traj["constant_boundary"][:, 0],
            "varying_boundary": traj["varying_boundary"],
        }
    ).to(device)
    c_scalar = traj["calendar"].to(device)
    return c_grid, c_scalar


# --------------------------------------------------------------------------- #
# Channel bookkeeping
# --------------------------------------------------------------------------- #
def _channel_names(model_cfg) -> list[str]:
    sfc = list(model_cfg.surface_variables)
    ua = list(model_cfg.get("upper_air_variables", []) or [])
    lev = list(model_cfg.get("levels", []) or [])
    dg = list(model_cfg.get("diagnostic_variables", []) or [])
    oc = list(model_cfg.get("ocean_state_variables", []) or [])
    layout = str(model_cfg.get("channel_layout", "v2"))
    if layout == "fork":
        ua_names = [f"{v}@{l}" for v in ua for l in lev]
        return sfc + ua_names + dg + oc
    # v1 / v2: [surface | diag | upper-air level-major, level axis flipped]
    ua_names = [f"{v}@{l}" for l in reversed(lev) for v in ua]
    return sfc + dg + ua_names + oc


def _group_slices(wrapper) -> dict[str, slice]:
    lay = wrapper.state_layout()
    ns, nd = lay["nsurface"], lay["ndiagnostic"]
    nu = lay["n_upper_air"] * lay["nlevels"]
    if getattr(wrapper, "channel_layout", "v2") == "fork":
        return {"surface": slice(0, ns), "upper_air": slice(ns, ns + nu),
                "diagnostic": slice(ns + nu, ns + nu + nd)}
    return {"surface": slice(0, ns), "diagnostic": slice(ns, ns + nd),
            "upper_air": slice(ns + nd, ns + nd + nu)}


def _astd(v: torch.Tensor) -> torch.Tensor:
    """Spatial-anomaly std over the trailing (H, W) axes."""
    v = v.float()
    m = v.mean(dim=(-2, -1), keepdim=True)
    return (v - m).pow(2).mean(dim=(-2, -1)).sqrt()


def _amean(v: torch.Tensor) -> torch.Tensor:
    return v.float().mean(dim=(-2, -1))


def _slope(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-(..., channel) regression slope of ``a`` on ``b`` over (H, W)."""
    a = a.float(); b = b.float()
    am = a.mean(dim=(-2, -1), keepdim=True); bm = b.mean(dim=(-2, -1), keepdim=True)
    cov = ((a - am) * (b - bm)).mean(dim=(-2, -1))
    var = (b - bm).pow(2).mean(dim=(-2, -1)).clamp_min(1e-12)
    return cov / var


def _perturb_state(y: torch.Tensor, n_state: int, *, shrink: float | None = None,
                   offset: float | None = None) -> torch.Tensor:
    """Perturb the state channels ``[:n_state]`` of a ``(..., C, H, W)`` tensor."""
    y = y.clone()
    st = y[..., :n_state, :, :]
    if shrink is not None:
        m = st.mean(dim=(-2, -1), keepdim=True)
        st = m + shrink * (st - m)
    if offset is not None:
        st = st + offset
    y[..., :n_state, :, :] = st
    return y


# --------------------------------------------------------------------------- #
# Streaming rollouts for both scheduler families
# --------------------------------------------------------------------------- #
class _Stream:
    """Uniform ``init(init_y) -> state`` / ``step(state, k) -> (emitted, state)``.

    RSI uses its own ``stream_init``/``stream_step``; ERDM has none, so its
    ``sample_rollout`` loop is replicated verbatim (erdm.py sample_rollout).
    """

    def __init__(self, sched, model, c_grid_traj, c_scalar_traj, num_steps):
        self.s, self.m = sched, model
        self.cg, self.cs, self.n = c_grid_traj, c_scalar_traj, num_steps
        self.family = "rsi" if hasattr(sched, "stream_init") else "erdm"

    def init(self, init_y):
        s = self.s
        if self.family == "rsi":
            return s.stream_init(init_y)
        init_y = s.pad_state(init_y)
        b = init_y.shape[0]
        sigma0 = s.sigma_schedule(torch.zeros(b, device=init_y.device))
        eps_win = s.temporal_noise(init_y)
        return (init_y + s.w5(sigma0) * eps_win, eps_win[:, -1:])

    def windows(self, k):
        s = self.s
        cgw = s._gather_window(self.cg, k)
        csw = s._gather_window(self.cs, k)
        ocw = s._gather_window(self.cg, k + 1) if getattr(s, "nocean", 0) else None
        return cgw, csw, ocw

    def step(self, state, k):
        s = self.s
        cgw, csw, ocw = self.windows(k)
        if self.family == "rsi":
            return s.stream_step(self.m, state, cgw, csw, self.n, ocean_win=ocw)
        x_bar, eps_prev = state
        x_bar = s.sample_window(self.m, x_bar, cgw, csw, self.n, ocean_win=ocw)
        emitted = x_bar[:, 0]
        eps_prev = s.temporal_noise_next(eps_prev)
        x_bar = torch.cat([x_bar[:, 1:], eps_prev * s.sigma_max], dim=1)
        return emitted, (x_bar, eps_prev)

    @torch.no_grad()
    def run(self, state, k_start: int, K: int, seed: int):
        torch.manual_seed(seed)
        outs = []
        for k in range(k_start, k_start + K):
            emitted, state = self.step(state, k)
            outs.append(emitted)
        return torch.stack(outs, dim=1), state

    @staticmethod
    def perturb_state(state, n_state, **kw):
        x = _perturb_state(state[0], n_state, **kw)
        return (x, state[1])


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_readout(sched, model, wrapper, ds, ics, step, device, *, batch, family,
                  t_globals=(0.5, 0.0), offset=0.3, shrink=0.7, num_steps=2):
    W = int(sched.W)
    nocean = int(getattr(sched, "nocean", 0) or 0)
    res = {}
    for tg in t_globals:
        acc = {k: [] for k in ("std_ratio", "slope", "rmse", "level_gain",
                               "shrink_ratio", "y_astd")}
        for b0 in range(0, len(ics), batch):
            bics = ics[b0:b0 + batch]
            B = len(bics)
            if family == "rsi":
                win = _batch_windows(ds, bics, 0, W + 1, step)          # rows t..t+W
            else:
                win = _batch_windows(ds, bics, 1, W, step)              # rows t+1..t+W
            y = wrapper.pack_window_state(_to(win, device)).float()
            n_state = y.shape[2]
            cg, cs = _forcing_traj(ds, wrapper, bics, W + 2, step, device)
            y = sched.pad_state(y)
            if nocean:
                if family == "rsi":
                    truth = sched.ocean_truth(cg[:, : W + 1], y.shape[-2:], expect=W + 1)
                else:
                    truth = sched.ocean_truth(cg[:, 1 : W + 1], y.shape[-2:])
                y[:, :, -nocean:] = truth.to(y.dtype)
            cgw, csw = cg[:, :W], cs[:, :W]

            def readout(y_):
                torch.manual_seed(1234)                 # identical latents per variant
                if family == "rsi":
                    tau = sched.local_time(torch.full((B,), float(tg), device=device))
                    anchors, targets = y_[:, :-1], y_[:, 1:]
                    z = sched.get_noise(targets)
                    x = sched.interpolant(anchors, targets, tau, z)
                    h1, _ = sched.heads(model, x, tau, cgw, csw)
                    return h1, targets
                sigma = sched.sigma_schedule(torch.full((B,), float(tg), device=device))
                eps = sched.temporal_noise(y_) if hasattr(sched, "temporal_noise") \
                    else torch.randn_like(y_)
                x_bar = y_ + sched.w5(sigma) * eps
                D = sched.denoise(model, x_bar, sigma, cgw, csw)
                return D, y_

            h1, tgt = readout(y)
            acc["std_ratio"].append((_astd(h1) / _astd(tgt).clamp_min(1e-8)).mean(0).cpu())
            acc["slope"].append(_slope(h1, tgt).mean(0).cpu())
            acc["rmse"].append((h1 - tgt).float().pow(2).mean(dim=(-2, -1)).sqrt().mean(0).cpu())
            acc["y_astd"].append(_astd(tgt).mean(0).cpu())
            h1_off, _ = readout(_perturb_state(y, n_state, offset=offset))
            acc["level_gain"].append(((_amean(h1_off) - _amean(h1)) / offset).mean(0).cpu())
            h1_sh, _ = readout(_perturb_state(y, n_state, shrink=shrink))
            acc["shrink_ratio"].append((_astd(h1_sh) / _astd(h1).clamp_min(1e-8)).mean(0).cpu())
            log.info(f"[readout t={tg}] ICs {bics} done")
        res[f"t{tg}"] = {k: torch.stack(v).mean(0) for k, v in acc.items()}   # (W, C)
    res["meta"] = {"t_globals": list(t_globals), "offset": offset, "shrink": shrink,
                   "family": family, "n_ics": len(ics)}
    return res


@torch.no_grad()
def probe_cascade(sched, model, wrapper, ds, ics, step, device, *, batch, family,
                  rolls, num_steps, trace_path=None):
    W = int(sched.W)
    nocean = int(getattr(sched, "nocean", 0) or 0)
    emit_std, emit_mean, truth_std, truth_mean, rmse = [], [], [], [], []
    for b0 in range(0, len(ics), batch):
        bics = ics[b0:b0 + batch]
        if family == "rsi":
            win = _batch_windows(ds, bics, 0, W + 1, step)
        else:
            win = _batch_windows(ds, bics, 1, W, step)
        init_y = wrapper.pack_window_state(_to(win, device)).float()
        traj_len = W + rolls - 1 + int(bool(nocean))
        cg, cs = _forcing_traj(ds, wrapper, bics, traj_len, step, device)
        if trace_path:
            os.environ["RSI_TRACE_PATH"] = f"{trace_path}.ics{b0}.pt"
            if hasattr(sched, "_trace_buf"):
                sched._trace_buf = None
        e_std, e_mean, t_std, t_mean, r = [], [], [], [], []
        t0 = time.time()
        for k0, x_k in sched.sample_rollout_generator(model, init_y, cg, cs,
                                                      horizon=rolls, num_steps=num_steps):
            k = k0 + 1
            em = sched.strip_ocean(x_k) if nocean else x_k
            tr = wrapper.pack_window_state(_to(_batch_windows(ds, bics, k, 1, step), device))[:, 0].float()
            e_std.append(_astd(em).mean(0).cpu()); e_mean.append(_amean(em).mean(0).cpu())
            t_std.append(_astd(tr).mean(0).cpu()); t_mean.append(_amean(tr).mean(0).cpu())
            r.append((em - tr).pow(2).mean(dim=(-2, -1)).sqrt().mean(0).cpu())
            if k % 10 == 0 or k == 1:
                log.info(f"[cascade] ICs {bics} roll {k}/{rolls} ({time.time() - t0:.0f}s)")
        emit_std.append(torch.stack(e_std)); emit_mean.append(torch.stack(e_mean))
        truth_std.append(torch.stack(t_std)); truth_mean.append(torch.stack(t_mean))
        rmse.append(torch.stack(r))
    if trace_path:
        os.environ.pop("RSI_TRACE_PATH", None)
    out = {
        "emit_std": torch.stack(emit_std).mean(0), "emit_mean": torch.stack(emit_mean).mean(0),
        "truth_std": torch.stack(truth_std).mean(0), "truth_mean": torch.stack(truth_mean).mean(0),
        "rmse": torch.stack(rmse).mean(0),
        "meta": {"rolls": rolls, "family": family, "n_ics": len(ics), "batch": batch},
    }
    if trace_path:
        parts = [torch.load(f"{trace_path}.ics{b0}.pt") for b0 in range(0, len(ics), batch)
                 if os.path.exists(f"{trace_path}.ics{b0}.pt")]
        if parts:
            out["trace"] = {k: torch.stack([p[k] for p in parts]).mean(0) for k in parts[0]}
    return out


@torch.no_grad()
def probe_flush(sched, model, wrapper, ds, ics, step, device, *, batch, family, rolls,
                k0, shrink, offset, num_steps, seed=0):
    W = int(sched.W)
    nocean = int(getattr(sched, "nocean", 0) or 0)
    groups = _group_slices(wrapper)
    results = {}
    for b0 in range(0, len(ics), batch):
        bics = ics[b0:b0 + batch]
        if family == "rsi":
            win = _batch_windows(ds, bics, 0, W + 1, step)
        else:
            win = _batch_windows(ds, bics, 1, W, step)
        init_y = wrapper.pack_window_state(_to(win, device)).float()
        n_state = init_y.shape[2]
        traj_len = W + k0 + rolls + 1 + int(bool(nocean))
        cg, cs = _forcing_traj(ds, wrapper, bics, traj_len, step, device)
        st = _Stream(sched, model, cg, cs, num_steps)

        def grp_rms(d: torch.Tensor) -> torch.Tensor:     # d: (B, K, C, H, W) -> (K, G)
            return torch.stack([d[:, :, sl].float().pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
                                for sl in groups.values()], dim=1).cpu()

        def denom(p: torch.Tensor) -> torch.Tensor:        # p: (B, C, H, W) -> (G,)
            return torch.stack([p[:, sl].float().pow(2).mean().sqrt() for sl in groups.values()]).cpu()

        case = {}
        # ---- from the true IC window ------------------------------------
        y_sh = _perturb_state(init_y, n_state, shrink=shrink)
        y_of = _perturb_state(init_y, n_state, offset=offset)
        torch.manual_seed(seed); ref, _ = st.run(st.init(init_y), 0, rolls, seed + 1)
        torch.manual_seed(seed + 7); ref2, _ = st.run(st.init(init_y), 0, rolls, seed + 8)
        torch.manual_seed(seed); psh, _ = st.run(st.init(y_sh), 0, rolls, seed + 1)
        torch.manual_seed(seed); pof, _ = st.run(st.init(y_of), 0, rolls, seed + 1)
        strip = (lambda v: v[:, :, :n_state]) if nocean else (lambda v: v)
        d_sh = denom((y_sh - init_y)[:, -1, :n_state]); d_of = denom((y_of - init_y)[:, -1, :n_state])
        case["ic"] = {
            "shrink": grp_rms(strip(psh - ref)) / d_sh, "offset": grp_rms(strip(pof - ref)) / d_of,
            "floor_shrink": grp_rms(strip(ref2 - ref)) / d_sh, "floor_offset": grp_rms(strip(ref2 - ref)) / d_of,
        }
        log.info(f"[flush ic] ICs {bics} done")
        # ---- from the free-run window after k0 rolls -----------------------
        torch.manual_seed(seed); _, state40 = st.run(st.init(init_y), 0, k0, seed + 1)
        x40 = state40[0]
        sh40 = _Stream.perturb_state(state40, n_state, shrink=shrink)
        of40 = _Stream.perturb_state(state40, n_state, offset=offset)
        ref, _ = st.run(state40, k0, rolls, seed + 11)
        ref2, _ = st.run(state40, k0, rolls, seed + 12)
        psh, _ = st.run(sh40, k0, rolls, seed + 11)
        pof, _ = st.run(of40, k0, rolls, seed + 11)
        d_sh = denom((sh40[0] - x40)[:, -1, :n_state]); d_of = denom((of40[0] - x40)[:, -1, :n_state])
        case["free"] = {
            "shrink": grp_rms(strip(psh - ref)) / d_sh, "offset": grp_rms(strip(pof - ref)) / d_of,
            "floor_shrink": grp_rms(strip(ref2 - ref)) / d_sh, "floor_offset": grp_rms(strip(ref2 - ref)) / d_of,
            "window_astd_ratio": (_astd(x40[:, :, :n_state]).mean(0) /
                                  _astd(init_y[:, -1:, :n_state]).mean(0).clamp_min(1e-8)).cpu(),
        }
        log.info(f"[flush free k0={k0}] ICs {bics} done")
        results[f"ics{b0}"] = case
    # average over IC batches
    out = {"groups": list(groups.keys()), "meta": {"rolls": rolls, "k0": k0, "shrink": shrink,
                                                   "offset": offset, "family": family}}
    for c in ("ic", "free"):
        out[c] = {}
        for k in results[next(iter(results))][c]:
            out[c][k] = torch.stack([results[b][c][k] for b in results]).mean(0)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _as_list(v) -> list[str]:
    """Accept a Hydra list (``[a,b]``), a quoted comma string, or a scalar."""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "ListConfig":
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _resolve_ics(raw_ds, ic_dates: list[str], log) -> list[int]:
    from inference import _full_time_coord, resolve_init_schedule  # noqa: E402

    times = _full_time_coord(raw_ds)
    rows = []
    for d in ic_dates:
        y, m, dd = (int(x) for x in d.split("-"))
        matches = resolve_init_schedule(times, months=[m], days=[dd], hours=[0], years=[y])
        if len(matches) != 1:
            raise ValueError(f"ic_date {d} resolved to {len(matches)} rows")
        rows.append(int(matches[0]))
        log.info(f"ic_date {d} -> global row {matches[0]} ({times[matches[0]]})")
    return rows


def _summarize(name, res, names, groups):
    s = {}
    if name == "readout":
        for tk, d in res.items():
            if tk == "meta":
                continue
            W = d["std_ratio"].shape[0]
            s[tk] = {
                "std_ratio_per_slot_mean": d["std_ratio"].mean(1).tolist(),
                "slope_per_slot_mean": d["slope"].mean(1).tolist(),
                "level_gain_per_slot_mean": d["level_gain"].mean(1).tolist(),
                "shrink_ratio_per_slot_mean": d["shrink_ratio"].mean(1).tolist(),
                "slotW_by_channel": {names[c]: {"std_ratio": float(d["std_ratio"][W - 1, c]),
                                                "slope": float(d["slope"][W - 1, c]),
                                                "level_gain": float(d["level_gain"][W - 1, c]),
                                                "shrink_ratio": float(d["shrink_ratio"][W - 1, c])}
                                     for c in range(min(len(names), d["std_ratio"].shape[1]))},
            }
    elif name == "cascade":
        ratio = res["emit_std"] / res["truth_std"].clamp_min(1e-8)     # (K, C)
        K = ratio.shape[0]
        s["emit_over_truth_astd_mean_over_channels"] = {
            str(k): float(ratio[k - 1].mean()) for k in (1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 120) if k <= K}
        s["emit_over_truth_astd_final_by_channel"] = {names[c]: float(ratio[-1, c]) for c in range(len(names))}
        if "trace" in res:
            tr = res["trace"]
            anc = tr["anchor_std"] / res["truth_std"][: tr["anchor_std"].shape[0]].clamp_min(1e-8)
            s["anchor_over_truth_astd_mean_over_channels"] = {
                str(k): float(anc[k - 1].mean()) for k in (1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 120) if k <= anc.shape[0]}
            gain = tr["anchor_std"][1:] / tr["anchor_std"][:-1].clamp_min(1e-8)
            s["anchor_per_roll_gain_mean_over_channels"] = {
                str(k): float(gain[k - 1].mean()) for k in (2, 5, 10, 20, 30, 40, 60, 80, 100, 120) if k - 1 < gain.shape[0]}
    elif name == "flush":
        for c in ("ic", "free"):
            s[c] = {}
            for k in ("shrink", "offset", "floor_shrink", "floor_offset"):
                v = res[c][k]                                            # (K, G)
                s[c][k] = {g: {str(r): float(v[r - 1, gi]) for r in (1, 6, 12, 24, 36, 48) if r <= v.shape[0]}
                           for gi, g in enumerate(groups)}
    return s


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from physicsnemo.distributed import DistributedManager

    from climate_eval_suite import _build_eval_dataset, _is_combined_model_cfg  # noqa: E402
    from inference import _apply_ema_weights  # noqa: E402
    from rollout import _build_stage, _load_group  # noqa: E402
    from train import _resolve_path  # noqa: E402
    from train_loop import adopt_ocean_contract, model_step_rows  # noqa: E402

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.backends.cuda.matmul.allow_tf32 = True

    pcfg = cfg.get("probes", None)
    if pcfg is None:
        raise ValueError("pass +probes.which=... (readout,cascade,flush) and +probes.out=...")
    which = _as_list(pcfg.which)
    out_dir = Path(_resolve_path(str(pcfg.out)))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _is_combined_model_cfg(cfg.model):
        raise ValueError("use a combined model config (model=amip_combined_*) so the "
                         "forecaster spec carries model/sampler/checkpoint")
    spec = cfg.model.forecaster
    f_model_cfg = _load_group("model", str(spec.model))
    cfg_eff = cfg.copy()
    with open_dict(cfg_eff):
        cfg_eff.model = f_model_cfg
    raw_ds, ds_cfg = _build_eval_dataset(cfg_eff, log)
    forecaster, sched = _build_stage(spec, device=device, log=log)
    weights = "raw"
    if bool(pcfg.get("use_ema", True)):
        f_path = Path(_resolve_path(str(spec.checkpoint)))
        stem = f_path.name.split(".")
        idx = stem[-2] if len(stem) >= 3 else None
        sib = f_path.parent / f"checkpoint.0.{idx}.pt"
        if sib.exists():
            blob = torch.load(sib, map_location=device, weights_only=False)
            meta = (blob.get("metadata") or {}) if isinstance(blob, dict) else {}
            weights = "ema" if _apply_ema_weights(forecaster, meta.get("ema"), logger=log) else "raw_ema_unavailable"
        else:
            log.warning(f"use_ema=True but no sibling {sib.name}; raw weights")
            weights = "raw_ema_unavailable"
    forecaster.eval()
    adopt_ocean_contract(sched, forecaster)
    family = "rsi" if hasattr(sched, "stream_init") else "erdm"
    step = model_step_rows(ds_cfg, raw_ds)
    names = _channel_names(f_model_cfg)
    groups = list(_group_slices(forecaster).keys())
    log.info(f"family={family} weights={weights} step={step} channels={len(names)} "
             f"sampler={type(sched).__name__} knobs: fresh_noise_scale="
             f"{getattr(sched, 'fresh_noise_scale', None)} final_denoise={getattr(sched, 'final_denoise', None)}")

    ic_dates = _as_list(pcfg.get("ic_dates", "1996-01-01"))
    ics = _resolve_ics(raw_ds, ic_dates, log)
    batch = int(pcfg.get("batch", 2))
    num_steps = int(pcfg.get("num_steps", 2))
    summary = {"family": family, "weights": weights, "checkpoint": str(spec.checkpoint),
               "sampler": str(spec.sampler), "ic_dates": ic_dates, "ic_rows": ics,
               "num_steps": num_steps, "channel_names": names, "groups": groups}
    t_all = time.time()
    if "readout" in which:
        t0 = time.time()
        res = probe_readout(sched, forecaster, forecaster, raw_ds, ics, step, device, batch=batch,
                            family=family, offset=float(pcfg.get("offset", 0.3)),
                            shrink=float(pcfg.get("shrink", 0.7)), num_steps=num_steps)
        torch.save(res, out_dir / "readout.pt")
        summary["readout"] = _summarize("readout", res, names, groups)
        log.info(f"readout done in {time.time() - t0:.0f}s: slot-mean std ratios at t=0.5 "
                 f"{summary['readout']['t0.5']['std_ratio_per_slot_mean']}")
    if "cascade" in which:
        t0 = time.time()
        res = probe_cascade(sched, forecaster, forecaster, raw_ds, ics, step, device, batch=batch,
                            family=family, rolls=int(pcfg.get("cascade_rolls", 120)),
                            num_steps=num_steps,
                            trace_path=str(out_dir / "trace") if family == "rsi" else None)
        torch.save(res, out_dir / "cascade.pt")
        summary["cascade"] = _summarize("cascade", res, names, groups)
        log.info(f"cascade done in {time.time() - t0:.0f}s: emitted/truth anomaly std "
                 f"{summary['cascade']['emit_over_truth_astd_mean_over_channels']}")
    if "flush" in which:
        t0 = time.time()
        res = probe_flush(sched, forecaster, forecaster, raw_ds, ics, step, device, batch=batch,
                          family=family, rolls=int(pcfg.get("flush_rolls", 36)),
                          k0=int(pcfg.get("flush_k0", 30)), shrink=float(pcfg.get("flush_shrink", 0.7)),
                          offset=float(pcfg.get("flush_offset", -0.3)), num_steps=num_steps,
                          seed=int(cfg.get("seed", 0)))
        torch.save(res, out_dir / "flush.pt")
        summary["flush"] = _summarize("flush", res, names, groups)
        log.info(f"flush done in {time.time() - t0:.0f}s: {json.dumps(summary['flush'])[:600]}")
    summary["seconds"] = round(time.time() - t_all, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
