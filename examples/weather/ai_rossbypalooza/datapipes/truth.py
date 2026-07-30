# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMERG daily-precip truth: yearly-store discovery and day lookup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .adapters import ReadRequest, _open_meta

logger = logging.getLogger(__name__)

#: array_key owner name for truth reads in the dataset's handle cache.
TRUTH_OWNER = "__truth__"

DayKey = tuple[int, int, int]  # (year, month, day) of the record's 00Z stamp


class ImergTruth:
    """Daily truth archive: one ``{root}/{YYYY}.zarr`` per year.

    Each record stamped 00Z on day X covers ``[X 00Z, X+1 00Z)`` (mm/day).
    Gaps (e.g. IMERG starts 2000-06-01) are simply absent from the day map,
    so pairs whose valid day is missing never enter the sample index.
    """

    def __init__(
        self, root: str | Path, *, var: str = "total_precipitation_24hr"
    ) -> None:
        self.root = Path(root)
        self.var = var
        self._day_lookup: dict[DayKey, tuple[int, int]] = {}
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None
        self._discovered = False

    def discover(self) -> None:
        stores = sorted(self.root.glob("*.zarr"))
        if not stores:
            raise ValueError(f"truth: no *.zarr stores under {self.root}")
        for store in stores:
            try:
                year = int(store.stem)
            except ValueError:
                logger.debug("truth: skipping non-year store %s", store)
                continue
            ds = _open_meta(store)
            try:
                if self.var not in ds.data_vars:
                    raise ValueError(
                        f"truth store {store} lacks variable '{self.var}'"
                    )
                if self.lat is None:
                    self.lat = ds["lat"].values.astype("float64")
                    self.lon = ds["lon"].values.astype("float64")
                for idx, t in enumerate(ds["time"].values):
                    key = (int(t.year), int(t.month), int(t.day))
                    if key in self._day_lookup:
                        logger.warning(
                            "truth: duplicate day %s in %s (keeping first)",
                            key,
                            store,
                        )
                        continue
                    self._day_lookup[key] = (year, idx)
            finally:
                ds.close()
        self._discovered = True

    def day_lookup(self) -> dict[DayKey, tuple[int, int]]:
        if not self._discovered:
            raise RuntimeError("truth: call discover() first")
        return self._day_lookup

    def plan(self, store_year: int, time_idx: int) -> ReadRequest:
        return ReadRequest((TRUTH_OWNER, store_year, self.var), (time_idx,))

    def store_path(self, store_year: int) -> Path:
        return self.root / f"{store_year}.zarr"


__all__ = ["ImergTruth", "TRUTH_OWNER", "DayKey"]
