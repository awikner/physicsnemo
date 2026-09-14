# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive sequence dataset over the packed PlaSim sim52 HDF5 tree.

Yields one training sample as

    state_in   [52, H, W]                 z-scored, the initial condition
    forcing    [n_steps, 8, H, W]         z-scored, the forcing at each target step
    state_tgt  [n_steps, 52, H, W]        z-scored targets for the AR rollout
    diag_tgt   [n_steps, 2, H, W]         PHYSICAL units, >= 0

Diagnostics are deliberately NOT normalized: the hurdle-Gamma head models them
in physical units (rescaled internally by a fixed per-channel constant), so the
likelihood stays interpretable and exceedance thresholds convert exactly.

Samples never straddle a year boundary - the packed files are one calendar year
each and consecutive years are contiguous in real time, but stitching across the
file seam would silently mix a leap and non-leap forcing cycle, so it is
disallowed.
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from channels import N_DIAGNOSTIC, N_FORCING, N_STATE


class PlasimSequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        n_steps: int = 1,
        years: list[int] | None = None,
        dt: int = 1,
    ):
        """
        Parameters
        ----------
        n_steps : int
            Number of autoregressive steps to supervise (1 = single-step).
        dt : int
            Stride in 6-hourly units between consecutive AR steps. 1 = 6 h.
        """
        self.root = root
        self.split = split
        self.n_steps = n_steps
        self.dt = dt

        state_dir = os.path.join(root, "state", split)
        files = sorted(glob.glob(os.path.join(state_dir, "*.h5")))
        if not files:
            raise FileNotFoundError(f"no packed years under {state_dir}")
        self.years = []
        for f in files:
            y = int(os.path.splitext(os.path.basename(f))[0])
            if years is None or y in years:
                self.years.append(y)
        if not self.years:
            raise ValueError(f"none of {years} present in {state_dir}")

        # Build a flat index of (year, t0) that fits a full rollout inside a year.
        self.index: list[tuple[int, int]] = []
        self.year_len: dict[int, int] = {}
        span = n_steps * dt
        for y in self.years:
            with h5py.File(self._path("state", y), "r") as f:
                T = f["fields"].shape[0]
            self.year_len[y] = T
            self.index.extend((y, t) for t in range(0, T - span))

        self._handles: dict[tuple[str, int], h5py.File] = {}
        self._load_stats()

    # ---------------------------------------------------------------- helpers

    def _path(self, kind: str, year: int) -> str:
        return os.path.join(self.root, kind, self.split, f"{year}.h5")

    def _load_stats(self):
        s = os.path.join(self.root, "stats")
        self.state_mean = np.load(os.path.join(s, "global_means.npy")).astype(np.float32)
        self.state_std = np.load(os.path.join(s, "global_stds.npy")).astype(np.float32)
        self.forcing_mean = np.load(
            os.path.join(s, "forcing_global_means.npy")
        ).astype(np.float32)
        self.forcing_std = np.load(
            os.path.join(s, "forcing_global_stds.npy")
        ).astype(np.float32)
        for name, a in (
            ("state_std", self.state_std),
            ("forcing_std", self.forcing_std),
        ):
            if not np.all(a > 0):
                bad = np.where(a.reshape(-1) <= 0)[0].tolist()
                raise ValueError(
                    f"{name} has non-positive entries at channels {bad}; a constant "
                    "channel would divide by zero (this is why `sic` was dropped)"
                )

    def _h(self, kind: str, year: int) -> h5py.File:
        # Opened lazily and cached per worker process; h5py handles are not
        # fork-safe, so they must not be created in __init__.
        key = (kind, year)
        if key not in self._handles:
            self._handles[key] = h5py.File(self._path(kind, year), "r")
        return self._handles[key]

    # ---------------------------------------------------------------- torch API

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        year, t0 = self.index[i]
        dt, n = self.dt, self.n_steps

        st = self._h("state", year)["fields"]
        fo = self._h("forcing", year)["fields"]
        dg = self._h("diagnostic", year)["fields"]

        state_in = np.asarray(st[t0], dtype=np.float32)
        ts = [t0 + (k + 1) * dt for k in range(n)]
        state_tgt = np.stack([np.asarray(st[t], dtype=np.float32) for t in ts])
        diag_tgt = np.stack([np.asarray(dg[t], dtype=np.float32) for t in ts])
        forcing = np.stack([np.asarray(fo[t], dtype=np.float32) for t in ts])

        state_in = (state_in - self.state_mean[0]) / self.state_std[0]
        state_tgt = (state_tgt - self.state_mean) / self.state_std
        forcing = (forcing - self.forcing_mean) / self.forcing_std

        assert state_in.shape[0] == N_STATE
        assert forcing.shape[1] == N_FORCING
        assert diag_tgt.shape[1] == N_DIAGNOSTIC

        return {
            "state_in": torch.from_numpy(state_in),
            "forcing": torch.from_numpy(forcing),
            "state_tgt": torch.from_numpy(state_tgt),
            "diag_tgt": torch.from_numpy(diag_tgt),
        }

    def __del__(self):
        for f in getattr(self, "_handles", {}).values():
            try:
                f.close()
            except Exception:
                pass
