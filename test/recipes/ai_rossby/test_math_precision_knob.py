# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The opt-in float32 math knobs for the diffusion recipe (2026-08-18).

``train.py`` has enabled TF32 and ``set_float32_matmul_precision("high")`` since
the Pangu benchmarks (~15% there); ``train_diffusion.py`` never did, and upstream
amip_v2's own ``train.py`` calls the same function. Measured on an A100-40GB at the
shipped v2 x_DDC geometry: **656 ms per training step at torch's default
``highest`` versus 365 ms at ``high``** — 1.80x, against a model whose upstream
already trains that way.

The knob is nonetheless OFF by default, because it changes a trained model's
numerics and that is the user's decision. These tests pin exactly that: default
changes nothing, each key is honoured when set, and a bad value is refused rather
than silently ignored.

Global torch state is saved and restored around every test — otherwise the first
test to enable TF32 would leave it on for every test that follows in the session,
which is precisely the kind of cross-test coupling that makes a suite lie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_RECIPE = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))

from train_loop import apply_math_precision  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_torch_math_state():
    saved = {
        "precision": torch.get_float32_matmul_precision(),
        "benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        saved["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
        saved["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(saved["precision"])
    torch.backends.cudnn.benchmark = saved["benchmark"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = saved["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_tf32"]


def _cfg(**kw):
    return OmegaConf.create(kw)


def test_default_changes_nothing():
    """The shipped configs set none of these, so a normal run must be untouched."""
    before = torch.get_float32_matmul_precision()
    applied = apply_math_precision(_cfg(amp="none", max_epochs=1))
    assert applied == {}
    assert torch.get_float32_matmul_precision() == before


def test_a_missing_training_block_is_not_an_error():
    assert apply_math_precision(None) == {}


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_each_valid_precision_is_applied_and_reported(precision):
    applied = apply_math_precision(_cfg(matmul_precision=precision))
    assert applied == {"matmul_precision": precision}
    assert torch.get_float32_matmul_precision() == precision


def test_an_unknown_precision_is_refused_by_name():
    """Silently ignoring it would leave a run slower than its config claims."""
    with pytest.raises(ValueError, match="matmul_precision='fast'"):
        apply_math_precision(_cfg(matmul_precision="fast"))


def test_cudnn_benchmark_is_honoured():
    applied = apply_math_precision(_cfg(cudnn_benchmark=True))
    assert applied == {"cudnn_benchmark": True}
    assert torch.backends.cudnn.benchmark is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 flags need CUDA")
def test_tf32_sets_both_matmul_and_cudnn():
    """Both, deliberately: conv TF32 and matmul TF32 are separate switches, and
    setting only one leaves half the model at full fp32."""
    applied = apply_math_precision(_cfg(allow_tf32=True))
    assert applied == {"allow_tf32": True}
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_all_three_together_report_together():
    applied = apply_math_precision(
        _cfg(matmul_precision="high", cudnn_benchmark=True)
    )
    assert applied["matmul_precision"] == "high"
    assert applied["cudnn_benchmark"] is True
