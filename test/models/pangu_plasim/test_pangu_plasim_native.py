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

"""Tests for the Phase 6 native Pangu_Plasim variants.

PanguPlasimLegacyNative / PanguPlasimNative are subclasses of the faithful
classes that differ only in their :class:`ModelMetaData` (advertising
CUDA-graph + AMP friendliness). Coverage:

* Forward parity with the faithful classes under matched RNG.
* :class:`ModelMetaData` flips ``cuda_graphs=True`` (which the faithful
  pair has as ``False``).
* Checkpoint roundtrip: save → ``Module.from_checkpoint`` → forward
  matches the pre-save output.
"""

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    import physicsnemo
    from physicsnemo.experimental.models.pangu_plasim import (
        PanguPlasim,
        PanguPlasimLegacy,
        PanguPlasimLegacyNative,
        PanguPlasimNative,
    )

from test import common

_SMOKE_KWARGS = dict(
    surface_variables=["t2m", "u10", "v10"],
    upper_air_variables=["t", "u", "v", "q", "z"],
    constant_boundary_variables=["lsm"],
    varying_boundary_variables=["rsdt"],
    levels=[200, 300, 500, 700, 850, 925, 1000, 1015],
    horizontal_resolution=[32, 64],
    patch_size=[2, 4, 4],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=64,
    window_size=[2, 4, 8],
    checkpointing=0,  # native variant's CUDA-graph mode requires no activation ckpts
)


def _make_inputs(device, batch_size=1):
    n_lat, n_lon = _SMOKE_KWARGS["horizontal_resolution"]
    n_levels = len(_SMOKE_KWARGS["levels"])
    n_surface = len(_SMOKE_KWARGS["surface_variables"])
    n_upper = len(_SMOKE_KWARGS["upper_air_variables"])
    n_const = len(_SMOKE_KWARGS["constant_boundary_variables"])
    n_vary = len(_SMOKE_KWARGS["varying_boundary_variables"])
    return (
        torch.randn(batch_size, n_surface, n_lat, n_lon, device=device),
        torch.randn(n_const, n_lat, n_lon, device=device),
        torch.randn(batch_size, n_vary, n_lat, n_lon, device=device),
        torch.randn(batch_size, n_upper, n_levels, n_lat, n_lon, device=device),
    )


@pytest.mark.parametrize(
    "native_cls,faithful_cls",
    [
        (PanguPlasimLegacyNative, PanguPlasimLegacy),
        (PanguPlasimNative, PanguPlasim),
    ],
    ids=["legacy", "vae"],
)
def test_native_metadata_advertises_cuda_graphs(native_cls, faithful_cls):
    """Faithful metadata has ``cuda_graphs=False`` (because of activation
    checkpointing); native metadata flips it to True."""
    torch.manual_seed(0)
    faithful = faithful_cls(**_SMOKE_KWARGS)
    torch.manual_seed(0)
    native = native_cls(**_SMOKE_KWARGS)

    assert faithful.meta.cuda_graphs is False
    assert native.meta.cuda_graphs is True
    assert native.meta.amp is True
    assert native.meta.bf16 is True
    assert native.meta.auto_grad is True


@pytest.mark.parametrize(
    "native_cls,faithful_cls",
    [
        (PanguPlasimLegacyNative, PanguPlasimLegacy),
        (PanguPlasimNative, PanguPlasim),
    ],
    ids=["legacy", "vae"],
)
def test_native_forward_matches_faithful(device, native_cls, faithful_cls):
    """With matched RNG, native forward output is bit-identical to the
    faithful one — they share constructor, layers, and forward. Only the
    MetaData differs."""
    torch.manual_seed(0)
    faithful = faithful_cls(**_SMOKE_KWARGS).to(device).eval()
    torch.manual_seed(0)
    native = native_cls(**_SMOKE_KWARGS).to(device).eval()

    # Sync weights so init noise doesn't leak in.
    native.load_state_dict(faithful.state_dict())

    inputs = _make_inputs(device, batch_size=1)
    torch.manual_seed(42)
    with torch.no_grad():
        out_faithful = faithful(*inputs)
    torch.manual_seed(42)
    with torch.no_grad():
        out_native = native(*inputs)

    assert common.compare_output(out_faithful, out_native, rtol=0, atol=0)

    del faithful, native
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "native_cls",
    [PanguPlasimLegacyNative, PanguPlasimNative],
    ids=["legacy", "vae"],
)
def test_native_checkpoint_roundtrip(device, native_cls, tmp_path):
    """``.mdlus`` checkpoint roundtrip preserves the native MetaData + forward."""
    torch.manual_seed(0)
    model = native_cls(**_SMOKE_KWARGS).to(device).eval()
    inputs = _make_inputs(device, batch_size=1)
    torch.manual_seed(42)
    with torch.no_grad():
        out_pre = model(*inputs)

    ckpt_path = tmp_path / f"{native_cls.__name__}_roundtrip.mdlus"
    model.save(str(ckpt_path))

    loaded = physicsnemo.Module.from_checkpoint(str(ckpt_path)).to(device).eval()
    assert loaded.meta.cuda_graphs is True, "native MetaData lost in checkpoint roundtrip"

    torch.manual_seed(42)
    with torch.no_grad():
        out_loaded = loaded(*inputs)

    assert common.compare_output(out_pre, out_loaded, rtol=1e-5, atol=1e-5)

    del model, loaded
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
