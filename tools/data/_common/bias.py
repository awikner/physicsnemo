# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse a directory of bias ``.npy`` files into per-variable level/hour arrays.

PanguWeather's bias directories use a flat filename convention:

* ``{var}_bias.npy`` — annual mean bias (no time-of-day dim).
* ``{var}_bias_{0,6,12,18}z.npy`` — diurnal-cycle annual bias at hour `H` UTC.
* For level-varying vars: ``{var}_{level}_bias[_{H}z].npy``, where ``level`` is
  the float value (e.g. ``5000.0`` Pa, ``0.0383`` sigma) from the source data.

This module exposes :func:`scan_bias_dir` to enumerate the directory once and
:func:`load_bias_arrays` to fan the per-file reads out across a process pool.
The output is a per-variable mapping that the climatology+bias Zarr converter
turns into the unified schema.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# {varname}[_{level}]_bias[_{H}z].npy — varname may contain underscores.
# We anchor on `_bias` so we don't need to enumerate per-dataset varnames here;
# the level token (if present) is the rightmost numeric chunk immediately
# preceding `_bias`. The hour suffix (if present) is `_{0,6,12,18}z` after.
_BIAS_FILE_RE = re.compile(
    r"^(?P<core>.+?)_bias(?:_(?P<hour>0|6|12|18)z)?\.npy$"
)
_LEVEL_TRAILER_RE = re.compile(r"^(?P<var>.+)_(?P<level>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


@dataclass
class BiasFileSpec:
    """Description of a single bias .npy file in a bias directory."""

    path: Path
    var: str
    level: Optional[float]  # None for surface vars
    hour: Optional[int]  # None for annual mean; 0/6/12/18 for diurnal-cycle.


@dataclass
class VariableBiasGroup:
    """Per-variable bookkeeping of which bias files exist."""

    var: str
    levels: list[float] = field(default_factory=list)  # empty for surface vars
    annual: dict[Optional[float], Path] = field(default_factory=dict)  # level → path
    diurnal: dict[tuple[int, Optional[float]], Path] = field(default_factory=dict)


def parse_bias_filename(path: Path) -> Optional[BiasFileSpec]:
    """Parse a bias filename. Returns ``None`` if it doesn't match the schema."""
    m = _BIAS_FILE_RE.match(path.name)
    if m is None:
        return None
    core = m.group("core")
    hour = int(m.group("hour")) if m.group("hour") else None

    # Try to split off a trailing level token from `core`. Multi-token var names
    # like `pr_6h` end up with no level token because `_6h` doesn't parse as a
    # numeric level (the "h" suffix). Sigma-level vars look like `ta_0.0383`;
    # pressure-level vars look like `zg_5000.0`.
    lm = _LEVEL_TRAILER_RE.match(core)
    if lm is not None:
        var = lm.group("var")
        try:
            level = float(lm.group("level"))
        except ValueError:
            var, level = core, None
    else:
        var, level = core, None

    return BiasFileSpec(path=path, var=var, level=level, hour=hour)


def scan_bias_dir(bias_dir: Path) -> dict[str, VariableBiasGroup]:
    """Walk ``bias_dir`` and group .npy files by variable.

    Returns
    -------
    dict[str, VariableBiasGroup]
        Mapping ``var_name → group``. ``group.levels`` is the sorted list of
        levels for which we have any bias file; surface vars get an empty list.
    """
    groups: Dict[str, VariableBiasGroup] = {}
    for path in sorted(bias_dir.iterdir()):
        if not path.name.endswith(".npy"):
            continue
        spec = parse_bias_filename(path)
        if spec is None:
            logger.warning("skipping non-bias .npy file %s", path)
            continue
        g = groups.setdefault(spec.var, VariableBiasGroup(var=spec.var))
        if spec.level is not None and spec.level not in g.levels:
            g.levels.append(spec.level)
        if spec.hour is None:
            g.annual[spec.level] = spec.path
        else:
            g.diurnal[(spec.hour, spec.level)] = spec.path
    for g in groups.values():
        g.levels.sort()
    return groups


def _load_one(path: Path) -> np.ndarray:
    """Worker target: load a single .npy as float32."""
    return np.load(path).astype("float32", copy=False)


def load_bias_arrays(
    paths: list[Path],
    *,
    max_workers: int,
) -> dict[Path, np.ndarray]:
    """Load a list of bias .npy paths in parallel, return ``{path: array}``."""
    if max_workers <= 1 or len(paths) <= 1:
        return {p: _load_one(p) for p in paths}
    out: Dict[Path, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_one, p): p for p in paths}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def collect_bias_paths(groups: dict[str, VariableBiasGroup]) -> List[Path]:
    """Flatten the per-var groups into a single list of paths for the loader."""
    paths: list[Path] = []
    for g in groups.values():
        paths.extend(g.annual.values())
        paths.extend(g.diurnal.values())
    return paths
