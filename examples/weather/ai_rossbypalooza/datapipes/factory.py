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

"""Hydra-config -> :class:`HindcastMixtureDataset` factory.

Consumes the ``dataset`` config group (see
``conf/dataset/hindcast_derecho.yaml``): a master channel list, a truth
root, normalization store paths, and a list of experts (name, schema, root,
precip spec, optional variable exclusions), plus per-split ``train:`` /
``val:`` blocks (years, init_months, lead_days, min_experts).
"""

from __future__ import annotations

from typing import Any, Mapping

from .adapters import build_adapter
from .dataset import HindcastMixtureDataset
from .precip import PrecipSpec
from .stats import ChannelStats
from .truth import ImergTruth
from .variables import ChannelLayout


def _as_dict(cfg: Any) -> Mapping:
    """Accept plain dicts or OmegaConf nodes without importing omegaconf."""
    if hasattr(cfg, "items"):
        return cfg
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError as exc:  # pragma: no cover
        raise TypeError(f"unsupported config type {type(cfg)}") from exc


def build_dataset(cfg_dataset: Any, split: str) -> HindcastMixtureDataset:
    """Build the dataset for ``split`` in ``{"train", "val"}``."""
    cfg = _as_dict(cfg_dataset)
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if split not in cfg or cfg[split] is None:
        raise KeyError(f"dataset config has no '{split}' block")
    block = _as_dict(cfg[split])

    layout = ChannelLayout(list(cfg["master_channels"]))
    norm = _as_dict(cfg["normalization"])
    # Either one combined store under "dynamical_stats" (stat coord) or the
    # separate ERA5 _mean.zarr / _std.zarr pair.
    combined = norm.get("dynamical_stats")
    stats = ChannelStats(
        norm.get("dynamical_mean") or combined,
        norm.get("dynamical_std") or combined,
        norm["precip_stats"],
        layout,
    )

    experts = []
    for e in cfg["experts"]:
        e = _as_dict(e)
        precip = PrecipSpec(**_as_dict(e["precip"]))
        experts.append(
            build_adapter(
                str(e["name"]),
                str(e["schema"]),
                str(e["root"]),
                layout,
                precip,
                exclude_variables=tuple(e.get("exclude_variables") or ()),
                min_lead_day=e.get("min_lead_day"),
                max_lead_day=e.get("max_lead_day"),
            )
        )

    truth_cfg = _as_dict(cfg["truth"])
    truth = ImergTruth(
        str(truth_cfg["root"]),
        var=str(truth_cfg.get("var", "total_precipitation_24hr")),
    )

    loader_cfg = _as_dict(cfg.get("loader") or {})
    min_experts = block.get("min_experts", 1)
    if isinstance(min_experts, str) and min_experts != "all":
        min_experts = int(min_experts)
    return HindcastMixtureDataset(
        experts,
        truth,
        layout,
        stats,
        years=tuple(int(y) for y in block["years"]),
        exclude_years=tuple(int(y) for y in (block.get("exclude_years") or ())),
        init_months=[int(m) for m in block["init_months"]],
        lead_days=tuple(int(d) for d in block["lead_days"]),
        min_experts=min_experts,
        zarr_concurrency=int(loader_cfg.get("zarr_concurrency", 6)),
        nan_expert_threshold=float(
            loader_cfg.get("nan_expert_threshold", 0.5)
        ),
    )
