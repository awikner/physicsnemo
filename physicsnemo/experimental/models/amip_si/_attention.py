# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Opt-in reduced-precision attention for the amip_si backbones.

Why this exists — measured, not assumed. Nsight Systems on a GH200, at the shipped
v2 geometries (``benchmarks/.../bench_v2_training_step.py --nvtx``, 2026-08-18):

===========================  =============  =============
kernel group                 fp32 step      TF32 step
===========================  =============  =============
mem-efficient attention      43.6% / 46.2%  **69.7% / 70.9%**
cuBLAS GEMMs                 ~50%           ~10%
===========================  =============  =============

(ERDM / x_DDC.) TF32 collapsed the GEMMs — ERDM's four biggest went from 6416 ms
to 705 ms across the traced steps — and left the attention kernels *bit-for-bit as
slow*: 3488.8 ms fp32 versus 3510.7 ms TF32 for the backward alone. They cannot
respond to the TF32 flag because they are hand-written CUTLASS kernels
(``fmha_cutlassB_f32_aligned_64x64_k64_sm80``), not cuBLAS GEMMs. Worse, they are
**sm80** kernels: fp32 has no flash implementation, so PyTorch selects the
mem-efficient backend, whose fp32 path was built for Ampere and runs unchanged on
Hopper.

So after enabling TF32, roughly 70% of a training step is fp32 attention that
Hopper cannot accelerate. Running *only the attention* in bf16 reaches the sm90
flash kernels while every other tensor stays fp32.

This is OFF by default: it changes the numerics of a trained model, which is the
caller's decision, not this module's. Enable per-process::

    from physicsnemo.experimental.models.amip_si import set_attention_dtype
    set_attention_dtype("bf16")

or from the recipe with ``++training.attention_dtype=bf16``.

A module-level setting rather than a constructor kwarg, deliberately: there are
seven ``scaled_dot_product_attention`` call sites across ``dit``, ``rolling_dit``,
``erdm_unet``, ``_unet_blocks`` and ``layers/cross_attention``, and threading a
dtype through every block's ``__init__`` would touch far more code than the one
line each site actually needs. Use :func:`attention_dtype` (a context manager) in
tests so the setting cannot leak between them.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "attention_dtype",
    "get_attention_dtype",
    "maybe_disable_cudnn_sdp",
    "sdpa",
    "set_attention_dtype",
]

_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "none": None,
    "off": None,
}

_ATTENTION_DTYPE: Optional[torch.dtype] = None


def _resolve(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """Normalize to a reduced dtype or ``None``.

    fp32 resolves to ``None`` rather than to ``torch.float32``: "run attention in
    fp32" and "do nothing" are the same request, and keeping them distinct would
    add a pointless pair of casts on the default path.
    """
    if not isinstance(dtype, torch.dtype):
        if dtype is not None:
            key = str(dtype).lower()
            if key not in _DTYPES:
                raise ValueError(
                    f"attention dtype {dtype!r} is not one of {sorted(_DTYPES)}"
                )
            dtype = _DTYPES[key]
    return None if dtype is torch.float32 else dtype


def set_attention_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """Set the dtype attention runs in. ``None``/``fp32`` restores the default."""
    global _ATTENTION_DTYPE
    _ATTENTION_DTYPE = _resolve(dtype)
    return _ATTENTION_DTYPE


def get_attention_dtype() -> Optional[torch.dtype]:
    return _ATTENTION_DTYPE


@contextlib.contextmanager
def attention_dtype(dtype: Union[str, torch.dtype, None]):
    """Scoped version of :func:`set_attention_dtype`."""
    previous = get_attention_dtype()
    try:
        set_attention_dtype(dtype)
        yield get_attention_dtype()
    finally:
        set_attention_dtype(previous)


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention``, optionally in a reduced dtype.

    Returns in the query's original dtype, so a caller cannot tell the difference
    except in the low bits — the surrounding graph stays fp32.

    A no-op when unset, and also when the tensors are ALREADY reduced-precision:
    under bf16 autocast there is nothing to gain and a cast back to fp32 would
    silently upcast the rest of the block.
    """
    dtype = _ATTENTION_DTYPE
    if dtype is None or query.dtype in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, **kwargs
        )
    # An additive float mask has to follow the dtype; a bool mask must NOT be
    # cast, since bool means "where to attend" and 0.0/1.0 would mean a bias.
    if attn_mask is not None and attn_mask.is_floating_point():
        attn_mask = attn_mask.to(dtype)
    out = F.scaled_dot_product_attention(
        query.to(dtype), key.to(dtype), value.to(dtype), attn_mask=attn_mask, **kwargs
    )
    return out.to(query.dtype)


def maybe_disable_cudnn_sdp() -> bool:
    """Drop the cuDNN attention backend when ``AI_ROSSBY_NO_CUDNN_SDPA`` is set.

    **Why this knob exists.** On DeltaAI (GH200, cuDNN 9.20, torch 2.10+cu129)
    ``F.scaled_dot_product_attention`` raises at the v2 geometries under bf16::

        RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error:
                      No valid execution plans built.

    Measured 2026-08-19 with a standalone 20-line probe — no repo code involved —
    at ``(B, H, S, D) = (6, 16, 4050, 64)``, i.e. RollingDiT v2's spatial
    attention (dim 1024 / 16 heads over a 45x90 token grid):

    ===============  ======  ======
    backend          bf16    fp32
    ===============  ======  ======
    default          FAIL    ok
    cudnn            FAIL    FAIL
    flash            ok      n/a
    mem-efficient    ok      ok
    math             ok      ok
    ===============  ======  ======

    So cuDNN's attention is simply broken for these shapes on that platform, and
    torch's backend *priority* reaches for it first under bf16 — which is why
    fp32 runs fine and a bf16 run dies. There is no env var for this in torch
    2.10 (``TORCH_CUDNN_SDPA_ENABLED=0`` is inert; verified), so it has to go
    through the Python API.

    **Disabling it costs nothing here.** Per this module's own analysis above,
    the point of reduced-precision attention is to reach the sm90 *flash*
    kernels; cuDNN was never the target. With cuDNN dropped, torch selects
    flash — exactly the intended path.

    Left OFF by default rather than applied unconditionally: on a platform where
    cuDNN attention works it may be the fastest choice, and a hard crash that
    names this function is a better failure mode than a silent slowdown
    everywhere. The launchers that need it export the variable
    (``hpc/scripts/train_rsi_amip.sbatch``, the ``deltaai-smoke-test`` skill);
    see the gotcha in ``hpc/deltaai.md``.

    Returns
    -------
    bool
        Whether the backend was disabled.
    """
    flag = os.environ.get("AI_ROSSBY_NO_CUDNN_SDPA", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        logger.warning(
            "AI_ROSSBY_NO_CUDNN_SDPA is set but this torch has no "
            "torch.backends.cuda.enable_cudnn_sdp; leaving the backend alone."
        )
        return False
    torch.backends.cuda.enable_cudnn_sdp(False)
    logger.info(
        "cuDNN SDPA backend disabled (AI_ROSSBY_NO_CUDNN_SDPA); attention will "
        "use flash / mem-efficient. Needed on DeltaAI GH200 + cuDNN 9.20, where "
        "the cuDNN attention path raises 'No valid execution plans built' at the "
        "v2 geometries under bf16."
    )
    return True
