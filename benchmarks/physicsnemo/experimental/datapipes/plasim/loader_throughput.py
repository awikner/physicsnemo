# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ASV format-level throughput comparison: PanguWeather v2.0 per-timestep
HDF5 (one file per sample, named-dataset-per-variable) vs the ai-rossby
PLASIM Zarr (one chunked store covering all timesteps).

The two backends iterate the **same** sample range and read the **same**
variables, both wrapped in a :class:`torch.utils.data.DataLoader` with
matching worker / prefetch / pin-memory settings. This isolates the
underlying file format / library from sampler / normalization / model code.

Data assumptions (Delta-host paths set as benchmark-class attrs below; can
be overridden via env vars at run time so the benchmark scales to other
hosts):

* PanguWeather HDF5 source: ``/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data``
  (per-timestep ``{year}_{idx:04d}.h5`` archive). Path override:
  ``$PLASIM_BENCH_H5_DIR``.
* Zarr fixture: ``$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr`` (the
  Phase-2 smoke fixture; 120 timesteps of sim52 year 100).

If either source is missing the benchmark raises a clear error from
:meth:`setup_cache` so ``asv run`` skips cleanly without poisoning the
history with bogus numbers.

A ``__main__`` block at the bottom runs a one-shot comparison and prints a
table — useful for directional sanity checks without spinning up an ASV
environment.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Default paths. Override at run time via env vars.
# ---------------------------------------------------------------------------
_DEFAULT_H5_DIR = Path(
    "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data"
)
_DEFAULT_H5_YEAR = 100
_DEFAULT_H5_SAMPLES = 120  # matches the Zarr fixture (30 days at 6-hourly)

# Variable layout used by both readers — chosen to mirror
# SFNO_PLASIM_H5_DERECHO_5412.yaml so the benchmark touches the same
# byte volume that production training would.
_SURFACE_VARS = ["pl", "tas"]
_CONSTANT_VARS = ["lsm", "sg", "z0"]
_VARYING_VARS = ["sst", "rsdt", "sic"]
_DIAG_VARS = ["pr_6h"]
_SIGMA_VARS = ["ta", "ua", "va", "hus"]
_PRESSURE_VARS = ["zg"]
_SIGMA_LEVELS = [
    0.03830000013113022, 0.11910000443458557, 0.21085000783205032,
    0.3168500065803528, 0.4368000030517578, 0.5668000280857086,
    0.6993500888347626, 0.8233500719070435, 0.9240999817848206,
    0.983299970626831,
]
_PRESSURE_LEVELS = [
    20000.0, 25000.0, 30000.0, 40000.0, 50000.0, 60000.0,
    70000.0, 85000.0, 92500.0, 100000.0,
]


# ---------------------------------------------------------------------------
# PanguWeather-style per-timestep HDF5 reader.
# ---------------------------------------------------------------------------
def _level_key(group, prefix: str, level: float, rtol: float = 1e-3) -> str:
    candidates = []
    for name in group:
        if not name.startswith(prefix + "_"):
            continue
        try:
            val = float(name[len(prefix) + 1:])
        except ValueError:
            continue
        candidates.append((val, name))
    for val, name in candidates:
        if abs(val - level) <= 1e-6 + rtol * abs(level):
            return name
    raise KeyError(f"No dataset for {prefix} at level {level}")


class PanguH5Dataset(Dataset):
    """Minimal per-timestep HDF5 reader matching the PanguWeather v2.0
    PLASIM data layout. One file open per ``__getitem__``; variables are
    read by name from the ``input/`` group; multi-level variables (sigma +
    pressure) are stacked by numerically matching the level-suffix keys.

    This is a **benchmark-only** baseline used to isolate file-format
    throughput from sampler / normalization code. Production training on
    PanguWeather-format data should go through PanguWeather's own loader,
    which adds per-node caching and lead-time sampling not modeled here.
    """

    def __init__(
        self,
        h5_dir: Path,
        year: int = _DEFAULT_H5_YEAR,
        n_samples: int = _DEFAULT_H5_SAMPLES,
    ) -> None:
        super().__init__()
        self.h5_dir = h5_dir
        self.year = year
        # Walk the directory once and cache the per-index file paths.
        pattern = re.compile(rf"^{year}_(\d{{4}})\.h5$")
        files: list[tuple[int, Path]] = []
        for p in sorted(h5_dir.iterdir()):
            m = pattern.match(p.name)
            if m:
                files.append((int(m.group(1)), p))
        files.sort(key=lambda t: t[0])
        self.files = [p for _, p in files][:n_samples]
        if len(self.files) != n_samples:
            raise FileNotFoundError(
                f"Need {n_samples} files at {h5_dir} year {year}; "
                f"found {len(self.files)}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.files[index]
        with h5py.File(path, "r") as f:
            grp = f["input"]
            surface = np.stack([grp[v][...] for v in _SURFACE_VARS]).astype("float32")
            constant = np.stack([grp[v][...] for v in _CONSTANT_VARS]).astype("float32")
            varying = np.stack([grp[v][...] for v in _VARYING_VARS]).astype("float32")
            diag = np.stack([grp[v][...] for v in _DIAG_VARS]).astype("float32")
            sigma_cube = np.empty(
                (len(_SIGMA_VARS), len(_SIGMA_LEVELS), *grp[_SURFACE_VARS[0]].shape),
                dtype="float32",
            )
            for v_i, v in enumerate(_SIGMA_VARS):
                for k_i, lev in enumerate(_SIGMA_LEVELS):
                    sigma_cube[v_i, k_i] = grp[_level_key(grp, v, lev)][...]
            pressure_cube = np.empty(
                (len(_PRESSURE_VARS), len(_PRESSURE_LEVELS), *grp[_SURFACE_VARS[0]].shape),
                dtype="float32",
            )
            for v_i, v in enumerate(_PRESSURE_VARS):
                for k_i, lev in enumerate(_PRESSURE_LEVELS):
                    pressure_cube[v_i, k_i] = grp[_level_key(grp, v, lev)][...]
            upper_air = np.concatenate([sigma_cube, pressure_cube], axis=0)
        return {
            "surface_in": torch.from_numpy(surface),
            "constant_boundary": torch.from_numpy(constant),
            "varying_boundary": torch.from_numpy(varying),
            "upper_air_in": torch.from_numpy(upper_air),
            "diagnostic": torch.from_numpy(diag),
        }


# ---------------------------------------------------------------------------
# ai-rossby Zarr-backed dataset (reuses the prod implementation).
# ---------------------------------------------------------------------------
def _build_zarr_dataset(zarr_path: Path, n_samples: int):
    """Open the Zarr fixture, wrap in a thin shim that returns only ``[i]`` (no
    paired-target) so the per-sample byte volume matches the PanguH5Dataset."""
    from physicsnemo.experimental.datapipes.plasim import PlasimClimateDataset

    class _SingleStepView(Dataset):
        def __init__(self) -> None:
            self.inner = PlasimClimateDataset(zarr_path)
            assert len(self.inner) >= n_samples, (
                f"Zarr fixture has {len(self.inner)} timesteps, need {n_samples}"
            )
            self.n = n_samples

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            # PlasimClimateDataset's int-index returns a paired (t, t+1) sample
            # with target_* fields; drop them so byte-count matches PanguH5Dataset.
            full = self.inner[(i, 1)] if i + 1 < len(self.inner) else self.inner[(i - 1, 1)]
            return {
                k: full[k]
                for k in (
                    "surface_in",
                    "constant_boundary",
                    "varying_boundary",
                    "upper_air_in",
                    "diagnostic",
                )
            }

    return _SingleStepView()


# ---------------------------------------------------------------------------
# Shared timing helper used by ASV time_* methods AND the __main__ runner.
# ---------------------------------------------------------------------------
def _iterate_one_epoch(loader) -> float:
    """Return wall-time in seconds to iterate the loader once."""
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - t0


def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )


def _h5_dir() -> Path:
    return Path(os.environ.get("PLASIM_BENCH_H5_DIR", str(_DEFAULT_H5_DIR)))


def _zarr_path() -> Path:
    root = os.environ.get("AI_ROSSBY_TEST_DATA")
    if not root:
        raise RuntimeError("AI_ROSSBY_TEST_DATA is not set")
    # smoke_month_t1.zarr writes 1 timestep per chunk (the recommended chunking
    # for per-sample reads — matches a training data loader's access pattern).
    # smoke_month.zarr exists too but uses a 50-step chunk which forces zarr
    # to load 50 samples per read; that fixture is for testing chunk-level
    # behavior, not loader throughput.
    return Path(root) / "plasim" / "smoke_month_t1.zarr"


# ---------------------------------------------------------------------------
# ASV benchmark classes.
# ---------------------------------------------------------------------------
class PanguH5LoaderThroughput:
    """Per-timestep HDF5 (PanguWeather v2.0 format) loader throughput."""

    bench_params = {
        "num_workers": [0, 2, 4],
        "batch_size": [1, 4],
    }
    params = list(bench_params.values())
    param_names = list(bench_params.keys())
    timeout = 600

    def setup(self, num_workers: int, batch_size: int) -> None:
        self.dataset = PanguH5Dataset(_h5_dir())
        self.loader = _make_loader(self.dataset, batch_size, num_workers)
        # One warmup epoch to populate Lustre/NVMe cache; isolates format
        # access pattern from cold-cache filesystem behavior.
        _iterate_one_epoch(self.loader)

    def teardown(self, num_workers: int, batch_size: int) -> None:
        # Persistent workers don't shut down until the DataLoader is gced;
        # force it explicitly.
        del self.loader

    def time_iterate_warm(self, num_workers: int, batch_size: int) -> None:
        for _ in self.loader:
            pass


class ZarrLoaderThroughput:
    """ai-rossby Zarr-backed loader throughput."""

    bench_params = {
        "num_workers": [0, 2, 4],
        "batch_size": [1, 4],
    }
    params = list(bench_params.values())
    param_names = list(bench_params.keys())
    timeout = 600

    def setup(self, num_workers: int, batch_size: int) -> None:
        self.dataset = _build_zarr_dataset(_zarr_path(), _DEFAULT_H5_SAMPLES)
        self.loader = _make_loader(self.dataset, batch_size, num_workers)
        _iterate_one_epoch(self.loader)

    def teardown(self, num_workers: int, batch_size: int) -> None:
        del self.loader

    def time_iterate_warm(self, num_workers: int, batch_size: int) -> None:
        for _ in self.loader:
            pass


# ---------------------------------------------------------------------------
# Training-step variant: load + forward + backward + AdamW step. Decides
# whether the loader gap matters operationally (Step 1 of the plan).
# ---------------------------------------------------------------------------
_PRODUCTION_PANGU_PLASIM_KW = dict(
    surface_variables=_SURFACE_VARS,
    upper_air_variables=[*_SIGMA_VARS, *_PRESSURE_VARS],
    constant_boundary_variables=_CONSTANT_VARS,
    varying_boundary_variables=_VARYING_VARS,
    diagnostic_variables=_DIAG_VARS,
    levels=_SIGMA_LEVELS,  # length-only is what the model reads
    horizontal_resolution=[64, 128],
    patch_size=[2, 4, 4],
    # Full PanguWeather defaults (embed_dim=192, depths=(2,6,6,2), num_heads=(6,12,12,6)).
    depths=[2, 6, 6, 2],
    num_heads=[6, 12, 12, 6],
    embed_dim=192,
    window_size=[2, 4, 8],
)


def _build_production_model(device: torch.device):
    """Production-shape PanguPlasim on the PLASIM 64x128 grid."""
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.models.pangu_plasim import PanguPlasim
    return PanguPlasim(**_PRODUCTION_PANGU_PLASIM_KW).to(device)


def _build_zarr_pipeline(
    zarr_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """Zarr-backed datapipe wired exactly the way Phase 3's training recipe
    would. Returns the iterable and the produced sample format
    (raw model-input tensors already on `device`, normalized, NaN-filled).
    """
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.plasim import (
            PlasimClimateDatapipe,
            PlasimClimateDataset,
            PlasimNormalizer,
        )
    layout_ds = PlasimClimateDataset(zarr_path)
    stats_dir = Path(
        "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data"
    )
    normalizer = PlasimNormalizer.from_dataset(
        layout_ds,
        mean_path=stats_dir / "data_12-132_mean_sigma.nc",
        std_path=stats_dir / "data_12-132_std_sigma.nc",
    )
    return PlasimClimateDatapipe(
        zarr_path,
        forecast_lead_times=[1],
        normalizer=normalizer,
        batch_size=batch_size,
        num_samples_per_epoch=_DEFAULT_H5_SAMPLES - 1,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        device=device,
        seed=0,
        shuffle=False,
    )


def _build_pangu_h5_pipeline(
    h5_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """PanguH5-backed loader for fair comparison. Same model inputs.

    Reuses :class:`PanguH5Dataset` for IO; normalization is applied inline on
    GPU using the same NetCDF stats (broadcasting works against the
    PanguH5Dataset's variable layout because variable order matches).
    """
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.plasim import (
            PlasimClimateDataset,
            PlasimNormalizer,
        )
    layout_ds = PlasimClimateDataset(_zarr_path())
    stats_dir = Path(
        "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data"
    )
    normalizer = PlasimNormalizer.from_dataset(
        layout_ds,
        mean_path=stats_dir / "data_12-132_mean_sigma.nc",
        std_path=stats_dir / "data_12-132_std_sigma.nc",
    ).to(device)
    loader = _make_loader(PanguH5Dataset(h5_dir), batch_size, num_workers)

    def _iter():
        for batch in loader:
            moved = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            moved["constant_boundary"] = torch.nan_to_num(
                moved["constant_boundary"], nan=0.0
            )
            moved["varying_boundary"] = torch.nan_to_num(
                moved["varying_boundary"], nan=0.0
            )
            yield normalizer(moved)

    class _Pipe:
        def __iter__(_self):
            return _iter()
        def __len__(_self):
            return len(loader)

    return _Pipe()


def _run_step_epoch(pipe, model, optimizer) -> tuple[float, float, int]:
    """Return (load_only_secs, step_secs, n_batches) for one full pass.

    ``load_only_secs`` times the loader-only iteration (no GPU work), and
    ``step_secs`` times the full load + forward + backward + AdamW step.
    Both are measured on the same loader to keep the cache state aligned.
    """
    # First pass: load-only (warm cache after the setup's warmup).
    t0 = time.perf_counter()
    n = 0
    for _ in pipe:
        n += 1
    load_only_secs = time.perf_counter() - t0

    # Second pass: load + forward + backward + step.
    t1 = time.perf_counter()
    n = 0
    for batch in pipe:
        optimizer.zero_grad()
        out = model(
            batch["surface_in"],
            batch["constant_boundary"],
            batch["varying_boundary"],
            batch["upper_air_in"],
        )
        loss = sum(t.sum() for t in out if t.requires_grad)
        loss.backward()
        optimizer.step()
        n += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_secs = time.perf_counter() - t1
    return load_only_secs, step_secs, n


def main_with_step(
    *,
    batch_size: int = 4,
    num_workers: int = 4,
    h5_dir: Optional[Path] = None,
    zarr_path: Optional[Path] = None,
) -> None:
    """Step-1 benchmark: compare load-only vs full step time for both backends.

    Decision rule (per the plan): if ``step / load > 5`` for both backends,
    the loader gap is not operationally meaningful and we stop. Otherwise we
    proceed to the optimization step.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Step benchmark requires CUDA; run on a Delta GPU node.")

    h5_dir = h5_dir or _h5_dir()
    zarr_path = zarr_path or _zarr_path()
    device = torch.device("cuda:0")

    print("=" * 78)
    print("PLASIM data loader: load-only vs load+forward+backward+step")
    print(f"  PanguPlasim production config: embed_dim=192, depths=(2,6,6,2), "
          f"num_heads=(6,12,12,6)")
    print(f"  Settings: num_workers={num_workers}, batch_size={batch_size}")
    print("=" * 78)

    rows: list[dict] = []

    # PanguH5 backend
    pipe = _build_pangu_h5_pipeline(
        h5_dir, batch_size=batch_size, num_workers=num_workers, device=device
    )
    model = _build_production_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Warmup
    for batch in pipe:
        out = model(
            batch["surface_in"], batch["constant_boundary"],
            batch["varying_boundary"], batch["upper_air_in"]
        )
        loss = sum(t.sum() for t in out if t.requires_grad)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break
    load_secs, step_secs, n = _run_step_epoch(pipe, model, optimizer)
    rows.append({"backend": "pangu_h5", "load_secs": load_secs,
                 "step_secs": step_secs, "n_batches": n})
    del model, optimizer, pipe
    torch.cuda.empty_cache()

    # Zarr backend
    pipe = _build_zarr_pipeline(
        zarr_path, batch_size=batch_size, num_workers=num_workers, device=device
    )
    model = _build_production_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for batch in pipe:
        out = model(
            batch["surface_in"], batch["constant_boundary"],
            batch["varying_boundary"], batch["upper_air_in"]
        )
        loss = sum(t.sum() for t in out if t.requires_grad)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break
    load_secs, step_secs, n = _run_step_epoch(pipe, model, optimizer)
    rows.append({"backend": "zarr", "load_secs": load_secs,
                 "step_secs": step_secs, "n_batches": n})
    del model, optimizer, pipe
    torch.cuda.empty_cache()

    # Report
    print(f"{'backend':12} {'n_batches':>10} {'load(s)':>10} {'step(s)':>10} "
          f"{'step/load':>10} {'batches/s':>12}")
    print("-" * 78)
    for r in rows:
        ratio = r["step_secs"] / r["load_secs"] if r["load_secs"] else float("inf")
        print(
            f"{r['backend']:12} {r['n_batches']:>10} "
            f"{r['load_secs']:>10.3f} {r['step_secs']:>10.3f} "
            f"{ratio:>9.2f}x {r['n_batches']/r['step_secs']:>12.2f}"
        )

    print()
    print("Decision rule (per pangu_plasim_reuse_plan.md follow-up):")
    print("  If step/load > 5 for BOTH backends -> loader is not the bottleneck")
    print("  -> Phase 2 IO format choice is operationally moot; move on to Phase 3.")
    ratios = [r["step_secs"] / r["load_secs"] for r in rows if r["load_secs"]]
    if all(r > 5.0 for r in ratios):
        print("  VERDICT: GPU-bound on both backends. Loader gap is moot.")
    else:
        print("  VERDICT: Loader-bound on at least one backend. Proceed to Step 2 "
              "(xarray -> raw-zarr swap).")


# ---------------------------------------------------------------------------
# Standalone runner — for quick directional sanity checks.
# ---------------------------------------------------------------------------
def _run_one(label: str, build_dataset, batch_size: int, num_workers: int) -> dict:
    dataset = build_dataset()
    loader = _make_loader(dataset, batch_size, num_workers)
    # Warmup epoch (cache warm).
    _iterate_one_epoch(loader)
    # Measured epoch.
    secs = _iterate_one_epoch(loader)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    del loader
    return {
        "label": label,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "samples": len(dataset),
        "secs": secs,
        "samples_per_sec": len(dataset) / secs,
        "batches_per_sec": n_batches / secs,
    }


def main(
    h5_dir: Optional[Path] = None,
    zarr_path: Optional[Path] = None,
    num_workers_grid: tuple[int, ...] = (0, 2, 4),
    batch_size_grid: tuple[int, ...] = (1, 4),
) -> None:
    h5_dir = h5_dir or _h5_dir()
    zarr_path = zarr_path or _zarr_path()

    print("=" * 78)
    print("PLASIM data loader throughput — warm cache, 120 samples (~30 days @ 6h)")
    print(f"  PanguWeather per-timestep HDF5: {h5_dir}")
    print(f"  ai-rossby Zarr:                  {zarr_path}")
    print("=" * 78)

    rows: list[dict] = []
    for nw in num_workers_grid:
        for bs in batch_size_grid:
            rows.append(
                _run_one(
                    "pangu_h5",
                    lambda: PanguH5Dataset(h5_dir),
                    bs,
                    nw,
                )
            )
            rows.append(
                _run_one(
                    "zarr",
                    lambda: _build_zarr_dataset(zarr_path, _DEFAULT_H5_SAMPLES),
                    bs,
                    nw,
                )
            )

    # Pretty-print table.
    print(
        f"{'backend':12} {'workers':>8} {'bs':>4} {'wall(s)':>10} "
        f"{'samples/s':>12} {'batches/s':>12}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['label']:12} {r['num_workers']:>8} {r['batch_size']:>4} "
            f"{r['secs']:>10.3f} {r['samples_per_sec']:>12.2f} "
            f"{r['batches_per_sec']:>12.2f}"
        )

    # Speedup summary: Zarr vs Pangu for each (workers, bs).
    print()
    print("Zarr speedup over PanguWeather per-timestep HDF5:")
    print(f"{'workers':>8} {'bs':>4} {'speedup':>10}")
    print("-" * 28)
    by_key = {(r["label"], r["num_workers"], r["batch_size"]): r for r in rows}
    for nw in num_workers_grid:
        for bs in batch_size_grid:
            pangu = by_key.get(("pangu_h5", nw, bs))
            zarr = by_key.get(("zarr", nw, bs))
            if pangu and zarr:
                speedup = pangu["secs"] / zarr["secs"]
                print(f"{nw:>8} {bs:>4} {speedup:>9.2f}x")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--with-step", action="store_true",
                   help="Run the load + model-step variant (requires CUDA).")
    p.add_argument("--zarr-path", type=Path, default=None)
    p.add_argument("--h5-dir", type=Path, default=None)
    args = p.parse_args()
    if args.with_step:
        main_with_step(h5_dir=args.h5_dir, zarr_path=args.zarr_path)
    else:
        main(h5_dir=args.h5_dir, zarr_path=args.zarr_path)
