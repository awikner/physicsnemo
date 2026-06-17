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

r"""Locally vendored copies of selected ``physicsnemo.nn.module`` classes,
patched for the ai-rossby ``pangu_plasim`` faithful flavor.

This sub-package is Phase C of
:doc:`../../../../../pangu_plasim_reuse_plan`. The vendored classes are
``FuserLayer``, ``Transformer3DBlock`` (transformer_layers.py),
``EarthAttention3D`` (attention_layers.py), ``Mlp`` (mlp_layers.py), and
``get_shift_window_mask`` (shift_window_mask.py). They are structurally and
API-equivalent to the upstream classes in ``physicsnemo.nn.module``, with the
following intentional patches:

1. **`get_shift_window_mask` — Issue
   `#1599 <https://github.com/NVIDIA/physicsnemo/issues/1599>`_ fix.** The
   longitude axis is *not* partitioned, restoring the cyclic-dateline
   attention behavior the PanguWeather v2.0 source intended. This restores
   the 9-region (3D) / 3-region (2D) mask the original Pangu-Weather paper
   describes. The upstream version produces 27 / 9 regions.

2. **`Transformer3DBlock` — ``vertical_windowing: bool`` kwarg.** Defaults to
   ``True`` (= upstream behavior). When ``False`` the pressure-level axis
   doesn't participate in the window shift, matching the PanguWeather v2.0
   no-vertical-shift configurations that the Pangu_Plasim models use.

3. **`EarthAttention3D` — ``use_sdpa: bool`` kwarg.** Defaults to ``False``
   (= upstream's explicit ``q @ k.T → softmax → attn @ v`` path). When
   ``True`` routes through :func:`torch.nn.functional.scaled_dot_product_attention`,
   matching the faster path the ai-rossby port uses by default.

This sub-package will be deleted once the upstream PRs for the three patches
land — at that point ``layers.py`` will import directly from
``physicsnemo.nn.module``.
"""

from .attention_layers import EarthAttention3D
from .mlp_layers import Mlp
from .shift_window_mask import get_shift_window_mask
from .transformer_layers import FuserLayer, Transformer3DBlock

__all__ = [
    "EarthAttention3D",
    "FuserLayer",
    "Mlp",
    "Transformer3DBlock",
    "get_shift_window_mask",
]
