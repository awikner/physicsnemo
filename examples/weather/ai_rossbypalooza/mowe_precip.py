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

"""MoWE precipitation gate: masked mixture of frozen weather experts.

Adapted from ``examples/weather/mixture_of_experts/mowe_model.py`` (a thin
DiT wrapper) with three changes for the ai-rossbypalooza Method-0 setup:

* **per-expert bias fields** — the gate emits ``2 E`` output channels
  (``E`` weight logits + ``E`` biases) and the mixture is
  ``P_hat = sum_i w_i * (P_i + b_i)`` instead of a single shared bias;
* **masked softmax** — a per-sample expert availability mask zeroes the
  weight of missing experts exactly (their bias vanishes with them), so
  one checkpoint works with any expert subset at inference;
* **mask planes** — the ``E`` mask bits are appended to the folded input
  channels as constant planes so the gate *sees* availability.

Lead time enters through DiT's conditioning scalar ``t`` (in whole days
here), exactly like the original recipe. As there, the mixture itself is
computed *outside* the model (:func:`mix`) so training/validation code can
log the gate-weight maps.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from physicsnemo.models.dit import DiT


class MoWEPrecipGate(DiT):
    """DiT gate over stacked expert blocks with per-expert weights + biases.

    Parameters
    ----------
    input_size : (H, W) of the common grid, e.g. ``(180, 360)``.
    in_channels : channels per expert block (``1 + C``: precip + predictors).
    n_experts : number of expert slots ``E`` (fixed by config; feeding fewer
        experts at inference = zeroed blocks + zeroed mask bits).
    patch_size : ViT patch size; must divide ``input_size`` (``(4, 4)`` for
        the 1-degree grid — 180 is not divisible by 8).
    noise_dim : optional conditioning-noise dimension for a future
        probabilistic variant (adds an ensemble axis exactly like the
        original MoWE); ``None`` = deterministic.

    Forward
    -------
    ``forward(x, mask, t, noise=None) -> (weights, biases)`` with
    ``x (B, E, in_channels, H, W)``, ``mask (B, E)`` in {0, 1} (>= 1 live
    expert per sample), ``t (B,)`` lead time in days. Outputs are
    ``(B, [ens,] E, H, W)``; weights are softmax over the expert axis with
    exact zeros at masked experts.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = (180, 360),
        *,
        in_channels: int,
        n_experts: int,
        patch_size: Union[int, Tuple[int, int]] = (4, 4),
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backend: str = "timm",
        layernorm_backend: str = "torch",
        noise_dim: Optional[int] = None,
        drop_path: float = 0.0,
    ):
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        patch = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        if size[0] % patch[0] or size[1] % patch[1]:
            raise ValueError(
                f"input_size {tuple(size)} must be divisible by "
                f"patch_size {tuple(patch)}"
            )
        # Stochastic depth: DiT wants one rate per block, so expand the
        # scalar into the usual linearly-increasing schedule 0 -> drop_path.
        if drop_path < 0 or drop_path >= 1:
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")
        drop_path_rates = (
            [drop_path * i / max(1, depth - 1) for i in range(depth)]
            if drop_path > 0
            else None
        )
        # E blocks folded into channels + E constant mask planes.
        net_in_channels = n_experts * in_channels + n_experts
        super().__init__(
            input_size=input_size,
            in_channels=net_in_channels,
            patch_size=patch_size,
            out_channels=2 * n_experts,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            condition_dim=noise_dim,
            drop_path_rates=drop_path_rates,
        )
        self.n_experts = n_experts
        self.block_channels = in_channels
        self.noise_dim = noise_dim

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, e, c, h, w = x.shape
        if e != self.n_experts or c != self.block_channels:
            raise ValueError(
                f"expected (B, {self.n_experts}, {self.block_channels}, H, W), "
                f"got {tuple(x.shape)}"
            )
        if (mask.sum(dim=1) < 1).any():
            raise ValueError("every sample needs at least one live expert")
        mask = mask.to(x.dtype)
        planes = mask.view(b, e, 1, 1).expand(b, e, h, w)
        net_in = torch.cat([x.reshape(b, e * c, h, w), planes], dim=1)

        if self.noise_dim:
            n_ens = noise.size(1)
            net_in = (
                net_in.unsqueeze(1)
                .expand(b, n_ens, *net_in.shape[1:])
                .reshape(b * n_ens, *net_in.shape[1:])
            )
            t = t.unsqueeze(1).expand(b, n_ens).reshape(-1) if t is not None else None
            noise = noise.reshape(b * n_ens, self.noise_dim)

        out = DiT.forward(self, net_in, t, noise)  # (B[*ens], 2E, H, W)

        if self.noise_dim:
            out = out.view(b, n_ens, 2 * e, h, w)
            logits, biases = out[:, :, :e], out[:, :, e:]
            mask_b = mask.view(b, 1, e, 1, 1)
            expert_dim = 2
        else:
            logits, biases = out[:, :e], out[:, e:]
            mask_b = mask.view(b, e, 1, 1)
            expert_dim = 1
        logits = logits.masked_fill(mask_b == 0, float("-inf"))
        weights = torch.softmax(logits, dim=expert_dim)
        return weights, biases


def mix(
    weights: torch.Tensor,
    biases: torch.Tensor,
    expert_precip: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``P_hat = sum_i w_i * (P_i + b_i)`` over the expert axis.

    ``expert_precip`` is each expert's precip channel in the mixing space,
    ``(B, E, H, W)``; ``weights`` / ``biases`` are the gate outputs,
    ``(B, [ens,] E, H, W)``. Returns ``(B, [ens,] H, W)``. Masked experts
    contribute nothing: their weight is exactly 0 and their bias is
    multiplied by it.

    ``mask (B, E)`` is an optional belt-and-braces guard. A missing expert's
    channel is zero-filled in z-space, which is NOT zero once inverted to
    mm/day (it is exp(mean) - eps ~ 0.7 mm/day of phantom rain), so passing
    the mask zeroes those entries explicitly rather than relying solely on
    the masked softmax.
    """
    if mask is not None:
        m = mask.to(expert_precip.dtype)
        expert_precip = expert_precip * m.view(
            m.shape[0], m.shape[1], *([1] * (expert_precip.ndim - 2))
        )
    if weights.ndim == 5:  # ensemble axis
        expert_precip = expert_precip.unsqueeze(1)
        return (weights * (expert_precip + biases)).sum(dim=2)
    return (weights * (expert_precip + biases)).sum(dim=1)


def expert_dropout(
    x: torch.Tensor,
    mask: torch.Tensor,
    p: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop live experts (training augmentation), keeping >= 1.

    Each live expert is dropped independently with probability ``p``; if a
    draw would kill every live expert of a sample, one originally-live
    expert (uniform among them) is kept. Returns the zeroed inputs and the
    new mask; inputs of dropped experts are zeroed to match the dataset's
    missing-expert convention.
    """
    if p <= 0.0:
        return x, mask
    b, e = mask.shape
    rand = torch.rand(b, e, generator=generator, device=mask.device)
    keep = ((rand >= p) & (mask > 0)).to(mask.dtype)
    dead = keep.sum(dim=1) == 0
    if dead.any():
        # Uniform choice among originally-live experts of each dead row.
        scores = torch.rand(b, e, generator=generator, device=mask.device)
        scores = scores * (mask > 0)  # zero for originally-missing experts
        pick = scores.argmax(dim=1)
        keep[dead, pick[dead]] = 1.0
    new_mask = mask * keep
    return x * new_mask.view(b, e, 1, 1, 1), new_mask
