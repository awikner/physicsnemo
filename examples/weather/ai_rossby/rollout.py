#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Streaming two-stage rollout: ERDM forecaster -> x_DDC downscaler.

Phase 12h.27 gave :class:`CombinedModule` ``windowed_init`` / ``windowed_step``
so a driver could emit one frame at a time and checkpoint between them, and left
the driver unwritten. This is it. ``conf/model/amip_combined.yaml`` already
describes the pairing (a forecaster and a downscaler, each with its own model
config, sampler and checkpoint); this consumes that shape.

Six things it does that ``inference.py`` structurally cannot:

1. **Two checkpoints, two schedulers.** ``inference.py`` builds exactly one model
   from ``cfg.model``. Both wrappers here are built from their own model configs
   and contract-checked against their artifacts before loading
   (``train_loop.assert_checkpoint_contract``) — a ``channel_layout`` mismatch
   moves no shape, so nothing else would notice.
2. **The resolution crossing.** ``inference.py`` takes output coords off the
   driving store, which for a cascade would label a 180x360 field with the
   forecaster's 45 latitudes. Coords come from a real high-resolution store
   instead (``rollout.highres_zarr``, defaulting to the dataset's boundary
   store). They are NOT synthesized from the grid shape: latitude order is not
   uniform across these archives (AMIP is S->N, ERA5 N->S), so inventing a
   coordinate is how a field ends up upside-down relative to its label. See
   docs/dev/context/lat-orientation-audit.md.
3. **Frame-at-a-time driving.** One ``windowed_step`` per emitted frame instead
   of a single ``sample_rollout`` over the whole horizon, so a multi-decade run
   never materialises the horizon.
4. **Mid-rollout resume.** ``(x_bar, eps_prev, step)`` is checkpointed every
   ``rollout.state_every`` frames and reloaded on restart.
   ``CombinedModule.windowed_step`` already refuses a rolling state whose channel
   count disagrees with the forecaster, which is what makes resuming from disk
   safe rather than merely possible.
5. **Per-stage sampler budgets.** ``windowed_step`` takes forecaster and
   downscaler ``num_steps`` separately; ``inference.py`` has one knob for both.
6. **Month-buffered output.** Frames accumulate and flush one file per calendar
   month through the existing :class:`AsyncForecastWriter`, rather than one file
   per initial condition.

Usage::

    python rollout.py model=amip_combined dataset=amip_dailyavg_coarse \
        +rollout.output_dir=./outputs/cascade \
        +rollout.ic_start=8 +rollout.horizon=120 \
        +rollout.forecaster_num_steps=2 +rollout.downscaler_num_steps=5
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from async_writer import AsyncForecastWriter  # noqa: E402
from train import _resolve_path, build_model  # noqa: E402
from train_diffusion import _build_dataset  # noqa: E402
from train_loop import (  # noqa: E402
    adopt_ocean_contract,
    assert_checkpoint_contract,
    model_step_rows,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si import CombinedModule

from physicsnemo import Module  # noqa: E402
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

_CONF = Path(__file__).resolve().parent / "conf"


def _load_group(group: str, stem: str) -> DictConfig:
    """Load one ``conf/<group>/<stem>.yaml``.

    The cascade needs TWO model configs and TWO samplers in one process, which
    Hydra's one-choice-per-group composition cannot express — hence reading the
    named files directly rather than through ``compose``.
    """
    path = _CONF / group / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no such config: {path}")
    return OmegaConf.load(path)


def _build_stage(spec: DictConfig, *, device, log) -> tuple[torch.nn.Module, object]:
    """Build one stage's wrapper + scheduler and load its checkpoint."""
    model_cfg = _load_group("model", str(spec.model))
    wrapper = build_model(model_cfg).to(device)
    ckpt = _resolve_path(str(spec.checkpoint))
    # Contract first, load second: the wrapper's packing came from the YAML, and
    # a mismatch against the artifact is shape-preserving.
    assert_checkpoint_contract(wrapper, ckpt, log=log)
    state = Module.from_checkpoint(ckpt).state_dict()
    wrapper.load_state_dict(state, strict=True)
    wrapper.eval()
    scheduler = hydra.utils.instantiate(_load_group("sampler", str(spec.sampler)))
    if hasattr(scheduler, "to"):
        scheduler = scheduler.to(device)
    log.info(
        f"{spec.model}: {sum(p.numel() for p in wrapper.parameters()) / 1e6:.2f}M "
        f"params, grid {tuple(wrapper.horizontal_resolution)}, "
        f"sampler {type(scheduler).__name__}"
    )
    return wrapper, scheduler


def _highres_coords(cfg: DictConfig, downscaler) -> tuple[np.ndarray, np.ndarray]:
    """lat/lon for the downscaler's grid, READ from a store — never synthesized.

    Latitude order differs across these archives (AMIP S->N, ERA5 N->S), so
    fabricating a coordinate from the grid shape is exactly how a field ends up
    upside-down relative to its label.
    """
    rcfg = cfg.get("rollout", {})
    src = rcfg.get("highres_zarr", None) or cfg.dataset.get("boundary_zarr_path", None)
    if not src:
        raise ValueError(
            "rollout needs high-resolution lat/lon for the downscaler's grid: set "
            "+rollout.highres_zarr=<a store at the downscaler's resolution>, or use "
            "a dataset config with boundary_zarr_path pointing at one. Deriving the "
            "coordinate from the grid shape is not done on purpose — row order is "
            "not uniform across these archives."
        )
    path = Path(_resolve_path(str(src)))
    if path.is_dir() and not str(path).endswith(".zarr"):
        years = sorted(p for p in path.glob("*.zarr"))
        if not years:
            raise FileNotFoundError(f"no per-year stores under {path}")
        path = years[0]
    ds = xr.open_zarr(path, consolidated=True, decode_times=False)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    want = tuple(downscaler.horizontal_resolution)
    if (len(lat), len(lon)) != want:
        raise ValueError(
            f"{path} is {(len(lat), len(lon))} but the downscaler emits {want}; "
            f"point rollout.highres_zarr at a store on the downscaler's grid"
        )
    return lat, lon


def _state_path(out_dir: Path) -> Path:
    return out_dir / "rollout_state.pt"


def _save_state(out_dir: Path, *, step: int, x_bar, eps_prev) -> None:
    """Checkpoint the rolling state so a killed job resumes mid-rollout.

    This is what the streaming API exists for; ``windowed_step`` already refuses
    a state whose channel count disagrees with the forecaster, so a resume across
    an ocean-contract change fails loudly instead of slicing silently.
    """
    tmp = _state_path(out_dir).with_suffix(".tmp")
    torch.save(
        {"step": int(step), "x_bar": x_bar.cpu(), "eps_prev": eps_prev.cpu()}, tmp
    )
    tmp.replace(_state_path(out_dir))          # atomic: never a half-written state


def _load_state(out_dir: Path, device):
    path = _state_path(out_dir)
    if not path.exists():
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    return int(blob["step"]), blob["x_bar"].to(device), blob["eps_prev"].to(device)


class _MonthBuffer:
    """Accumulate emitted frames and flush one file per calendar month.

    A multi-decade cascade would otherwise write one file per initial condition
    (``inference.py``'s unit) or one per frame — neither is a sensible object to
    hand a downstream analysis.
    """

    def __init__(self, *, writer, out_dir: Path, run_name: str, lat, lon, layout):
        self._writer = writer
        self._dir = out_dir
        self._run = run_name
        self._lat, self._lon = lat, lon
        self._layout = layout
        self._key: Optional[tuple[int, int]] = None
        self._frames: list[dict] = []
        self.paths: list[str] = []

    def add(self, time, fields: dict[str, torch.Tensor]) -> None:
        key = (int(time.year), int(time.month))
        if self._key is not None and key != self._key:
            self.flush()
        self._key = key
        self._frames.append({"time": time, **{k: v.cpu() for k, v in fields.items()}})

    def flush(self) -> None:
        if not self._frames or self._key is None:
            return
        year, month = self._key
        ds = self._to_dataset(self._frames)
        path = str(self._dir / f"{self._run}__{year:04d}{month:02d}.nc")
        self._writer.submit(path, ds)
        self.paths.append(path)
        self._frames = []
        self._key = None

    def _to_dataset(self, frames: list[dict]) -> xr.Dataset:
        times = [f["time"] for f in frames]
        coords = {
            "time": ("time", np.asarray(times)),
            "lat": ("lat", self._lat),
            "lon": ("lon", self._lon),
        }
        data_vars = {}
        surface = self._layout["surface_variables"]
        if surface:
            arr = np.stack([f["surface"].numpy() for f in frames])   # (T, C, H, W)
            coords["surface_var"] = ("surface_var", np.array(surface, dtype=object))
            data_vars["surface"] = (("time", "surface_var", "lat", "lon"), arr)
        diag = self._layout["diagnostic_variables"]
        if diag and "diagnostic" in frames[0]:
            arr = np.stack([f["diagnostic"].numpy() for f in frames])
            coords["diag_var"] = ("diag_var", np.array(diag, dtype=object))
            data_vars["diagnostic"] = (("time", "diag_var", "lat", "lon"), arr)
        upper = self._layout["upper_air_variables"]
        if upper and "upper_air" in frames[0]:
            arr = np.stack([f["upper_air"].numpy() for f in frames])  # (T, C, L, H, W)
            coords["upper_air_var"] = (
                "upper_air_var", np.array(upper, dtype=object)
            )
            coords["level"] = ("level", np.asarray(self._layout["levels"], "float32"))
            data_vars["upper_air"] = (
                ("time", "upper_air_var", "level", "lat", "lon"), arr
            )
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._layout["attrs"])


def _traj_windows(cfg, dataset, forecaster, scheduler, *, ic, horizon, step_size,
                  normalizer, device, log):
    """Pack the whole forcing trajectory once, to be sliced per step.

    Same construction ``inference.py``'s rolling branch uses, including the extra
    lookahead frame the predicted-ocean imposition needs (the ocean truth is the
    forcing window shifted one step forward).
    """
    from inference import _maybe_normalize, _stack_at_step

    W = int(scheduler.window_size)
    ocean_lookahead = int(bool(getattr(scheduler, "nocean", 0)))
    n_archive = int(getattr(dataset, "n_time", 0) or 0)
    if ocean_lookahead and n_archive and (
        ic + (W + horizon - 1) * step_size >= n_archive
    ):
        ocean_lookahead = 0
        log.warning("ocean lookahead disabled: the last roll runs past the archive")
    # Slot j = forcing at absolute step j (row ic + j * step_size): roll k's
    # window slot w holds state y_{k+w+1} and is conditioned on traj[k+w],
    # its lag-1 forcing — the forcing_lag=1 training alignment.
    traj_len = W + horizon - 1 + ocean_lookahead
    last_row = ic + (traj_len - 1) * step_size
    if n_archive and last_row >= n_archive:
        raise ValueError(
            f"rollout from ic={ic} needs store row {last_row} (forcing "
            f"trajectory of W={W} + horizon={horizon}) but the archive has "
            f"{n_archive} rows; use an earlier IC or a shorter horizon."
        )
    frames = [
        _maybe_normalize(
            normalizer,
            _stack_at_step(dataset, [ic + j * step_size], device),
        )
        for j in range(traj_len)
    ]
    stack = lambda key: torch.cat(  # noqa: E731
        [f[key].unsqueeze(1) for f in frames], dim=1
    )
    const = frames[0]["constant_boundary"]
    if const.dim() == 4:
        const = const.unsqueeze(1).expand(-1, traj_len, -1, -1, -1)
    c_grid = forecaster.pack_window_c_grid({
        "surface_in": stack("surface_in"),
        "constant_boundary": const,
        "varying_boundary": stack("varying_boundary"),
    })
    return c_grid, stack("calendar"), ocean_lookahead


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    log = PythonLogger("ai_rossby_rollout")

    rcfg = cfg.get("rollout", None)
    if rcfg is None:
        raise ValueError(
            "rollout.* config block missing; add +rollout.output_dir, "
            "+rollout.ic_start and +rollout.horizon on the command line."
        )
    missing = [k for k in ("output_dir", "ic_start", "horizon") if k not in rcfg]
    if missing:
        raise ValueError(f"rollout.* is missing required key(s): {', '.join(missing)}")

    combined_cfg = cfg.model
    for side in ("forecaster", "downscaler"):
        if side not in combined_cfg:
            raise ValueError(
                f"cfg.model has no '{side}' block — rollout.py drives a two-stage "
                f"cascade, so it needs a combined config (model=amip_combined), "
                f"not a single-model one"
            )

    out_dir = Path(_resolve_path(str(rcfg.output_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    forecaster, f_sched = _build_stage(combined_cfg.forecaster, device=dist.device, log=log)
    downscaler, d_sched = _build_stage(combined_cfg.downscaler, device=dist.device, log=log)
    adopt_ocean_contract(f_sched, forecaster)
    combined = CombinedModule(
        forecaster=forecaster, forecaster_scheduler=f_sched,
        downscaler=downscaler, downscaler_scheduler=d_sched,
    ).to(dist.device).eval()

    # The dataset is driven by the FORECASTER's contract, so hand _build_dataset a
    # cfg whose model block is the forecaster's — that reuses the whole
    # fill/normalize/route pipeline (and its varying-boundary subset) unchanged.
    f_model_cfg = _load_group("model", str(combined_cfg.forecaster.model))
    ds_cfg = OmegaConf.create(
        {"model": f_model_cfg, "dataset": cfg.dataset, "seed": cfg.get("seed", 0)}
    )
    dataset = _build_dataset(ds_cfg)
    if getattr(dataset, "forcing_pipeline", None) is not None:
        dataset.forcing_pipeline.assert_matches(forecaster, name="forecaster config")
    # NO further normalization here. _build_dataset attaches the forcing pipeline
    # as ds.transform with normalize_in_dataset=True, so samples arrive already
    # filled, normalized and scalar-routed. inference.py normalizes explicitly
    # because it builds its dataset WITHOUT the pipeline; copying that call here
    # would z-score twice, which changes no shape and quietly ruins the run.
    normalizer = None
    step_size = model_step_rows(ds_cfg, dataset)

    lat, lon = _highres_coords(cfg, downscaler)
    log.info(
        f"high-res coords from a store: {len(lat)}x{len(lon)}, "
        f"lat[0]={lat[0]:.2f} lat[-1]={lat[-1]:.2f} "
        f"({'S->N' if lat[0] < lat[-1] else 'N->S'})"
    )

    ic = int(rcfg.ic_start)
    horizon = int(rcfg.horizon)
    f_steps = rcfg.get("forecaster_num_steps", None)
    d_steps = rcfg.get("downscaler_num_steps", None)
    state_every = int(rcfg.get("state_every", 50))

    c_grid_traj, c_scalar_traj, ocean_lookahead = _traj_windows(
        cfg, dataset, forecaster, f_sched, ic=ic, horizon=horizon,
        step_size=step_size, normalizer=normalizer, device=dist.device, log=log,
    )

    resumed = _load_state(out_dir, dist.device)
    if resumed is not None:
        start_step, x_bar, eps_prev = resumed
        log.info(f"resuming at frame {start_step} from {_state_path(out_dir)}")
    else:
        from inference import _stack_window_initial

        # The oracle stack is the FUTURE window ending at ic + W (ERDM:
        # y_{1:W}); a data-coupled scheduler (RSI) asks for one more frame
        # and reaches back to the IC itself for its anchor y_0.
        n_init = int(getattr(f_sched, "init_frames", f_sched.window_size))
        init = _stack_window_initial(
            dataset, ic, n_init, dist.device, step_size=step_size,
            end_offset=int(f_sched.window_size))
        x_bar, eps_prev = combined.windowed_init(forecaster.pack_window_state(init))
        start_step = 0

    times = _frame_times(dataset, ic, horizon, step_size)
    layout = {
        "surface_variables": list(f_model_cfg.get("surface_variables", []) or []),
        "diagnostic_variables": list(f_model_cfg.get("diagnostic_variables", []) or []),
        "upper_air_variables": list(f_model_cfg.get("upper_air_variables", []) or []),
        "levels": [float(v) for v in (f_model_cfg.get("levels", []) or [])],
        "attrs": {
            "forecaster": str(combined_cfg.forecaster.model),
            "downscaler": str(combined_cfg.downscaler.model),
            "model_step_rows": int(step_size),
            "ic_index": ic,
            "note": "two-stage cascade; fields are on the DOWNSCALER's grid",
        },
    }

    with AsyncForecastWriter(
        max_in_flight=int(rcfg.get("writer_max_in_flight", 4)),
        num_workers=int(rcfg.get("writer_num_workers", 2)),
    ) as writer:
        buf = _MonthBuffer(
            writer=writer, out_dir=out_dir,
            run_name=str(cfg.get("run_name", "cascade")),
            lat=lat, lon=lon, layout=layout,
        )
        run_rollout(
            combined=combined, buf=buf, out_dir=out_dir, times=times,
            x_bar=x_bar, eps_prev=eps_prev,
            c_grid_traj=c_grid_traj, c_scalar_traj=c_scalar_traj,
            start_step=start_step, horizon=horizon,
            forecaster_num_steps=f_steps, downscaler_num_steps=d_steps,
            ocean_lookahead=bool(ocean_lookahead), state_every=state_every, log=log,
        )
    log.info(f"wrote {len(buf.paths)} monthly file(s) to {out_dir}")


def run_rollout(
    *,
    combined,
    buf,
    out_dir: Path,
    times,
    x_bar,
    eps_prev,
    c_grid_traj,
    c_scalar_traj,
    start_step: int,
    horizon: int,
    forecaster_num_steps=None,
    downscaler_num_steps=None,
    ocean_lookahead: bool = False,
    state_every: int = 50,
    log=None,
):
    """The streaming loop: one emitted frame per iteration.

    Split out of :func:`main` so it can be driven without Hydra, a dataset or
    real checkpoints — the loop is where the frame accounting, the ocean
    lookahead offset and the resume cadence live, and all three are easy to get
    subtly wrong.

    ``ocean_win`` is the forcing window one step FORWARD of the conditioning
    window: the predicted ocean channels are supervised against the boundary at
    each frame's own time, not at the time it was conditioned on.
    """
    log = log or logging.getLogger(__name__)
    sched = combined.forecaster_scheduler
    downscaler = combined.downscaler
    for k in range(start_step, horizon):
        ocean_win = (
            sched._gather_window(c_grid_traj, k + 1) if ocean_lookahead else None
        )
        y_high, x_bar, eps_prev = combined.windowed_step(
            x_bar, eps_prev,
            sched._gather_window(c_grid_traj, k),
            sched._gather_window(c_scalar_traj, k),
            forecaster_num_steps,
            ocean_win=ocean_win,
            downscaler_num_steps=downscaler_num_steps,
        )
        fields = downscaler.unpack_state(y_high)
        emitted = {"surface": fields["surface_in"][0]}
        if "upper_air_in" in fields:
            emitted["upper_air"] = fields["upper_air_in"][0]
        if "diagnostic" in fields:
            emitted["diagnostic"] = fields["diagnostic"][0]
        buf.add(times[k], emitted)
        if state_every and (k + 1) % state_every == 0:
            buf.flush()
            _save_state(out_dir, step=k + 1, x_bar=x_bar, eps_prev=eps_prev)
            log.info(f"frame {k + 1}/{horizon} — state checkpointed")
    buf.flush()
    _save_state(out_dir, step=horizon, x_bar=x_bar, eps_prev=eps_prev)
    return x_bar, eps_prev


def _frame_times(dataset, ic: int, horizon: int, step_size: int):
    """Valid time per emitted frame, from the archive's own time coord."""
    from inference import _full_time_coord

    times = _full_time_coord(dataset)
    out = []
    for k in range(horizon):
        idx = ic + (k + 1) * step_size
        if times is not None and idx < len(times):
            out.append(times[idx])
        else:
            out.append(np.datetime64("NaT"))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
