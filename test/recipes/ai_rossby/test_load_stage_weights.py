# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Which weights an evaluation actually scored must never be ambiguous.

Two silent failures motivate this:

* ``save_checkpoint`` writes the **live** weights to the ``.mdlus`` and the EMA
  shadow into ``metadata["ema"]`` of ``checkpoint.0.<epoch>.pt``, while the
  training loop validates **EMA-applied**. So loading a ``.mdlus`` on its own
  scores weights that no reported training metric describes. Upstream
  ``amip_v2`` does no EMA swap at inference, so both choices are defensible —
  but the choice has to be *recorded*, which is what ``weights_used`` is for.
* ``load_checkpoint`` defaults to the **latest** epoch. An evaluation labelled
  "epoch 10" that silently read epoch 14 looks completely normal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from inference import _apply_ema_weights  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self, fill=0.0):
        super().__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(fill)


def _ema_metadata(fill):
    """The exact shape ModelEMA.state_dict() persists."""
    shadow = _Tiny(fill)
    avg = {f"module.{k}": v for k, v in shadow.state_dict().items()}
    avg["n_averaged"] = torch.tensor(7)
    return {"avg_model_state": avg, "decay": 0.999,
            "warmup_epochs": 6, "steps_per_epoch": 100}


def test_ema_swap_replaces_the_live_weights():
    m = _Tiny(1.0)
    assert _apply_ema_weights(m, _ema_metadata(5.0)) is True
    assert torch.allclose(m.lin.weight, torch.full_like(m.lin.weight, 5.0)), \
        "EMA shadow was not copied onto the model"


def test_missing_ema_falls_back_and_reports_it():
    """The fallback must be visible, not silent: callers turn False into
    weights_used='raw_ema_unavailable'."""
    m = _Tiny(1.0)
    assert _apply_ema_weights(m, None) is False
    assert torch.allclose(m.lin.weight, torch.full_like(m.lin.weight, 1.0))


def test_raw_and_ema_weights_actually_differ():
    """Guards the whole point: if raw == EMA the distinction would be moot and
    a wrong default would be undetectable."""
    raw, ema = _Tiny(1.0), _Tiny(1.0)
    _apply_ema_weights(ema, _ema_metadata(5.0))
    assert not torch.allclose(raw.lin.weight, ema.lin.weight)


def test_accepts_a_bare_param_dict_without_the_module_prefix():
    m = _Tiny(1.0)
    shadow = {k: torch.full_like(v, 3.0) for k, v in _Tiny().state_dict().items()}
    assert _apply_ema_weights(m, {"avg_model_state": shadow}) is True
    assert torch.allclose(m.lin.weight, torch.full_like(m.lin.weight, 3.0))


def test_config_exposes_use_ema_and_checkpoint_epoch():
    """The knobs must exist in the shipped config, or a run cannot pin either."""
    import yaml

    cfg = yaml.safe_load(
        (_AI_ROSSBY_DIR / "conf" / "validation" / "eval_suite.yaml").read_text()
    )
    assert "use_ema" in cfg, sorted(cfg)
    assert "checkpoint_epoch" in cfg, sorted(cfg)
    assert cfg["checkpoint_epoch"] is None, "must default to latest, explicitly"
