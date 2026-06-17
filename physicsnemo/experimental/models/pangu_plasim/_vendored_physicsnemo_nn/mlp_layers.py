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

r"""Vendored copy of :class:`physicsnemo.nn.module.mlp_layers.Mlp` — the
``fc1 → act → drop → fc2 → drop`` block used by
:class:`~_vendored_physicsnemo_nn.transformer_layers.Transformer3DBlock`.

Identical to the upstream class — vendored here only because Phase C touches
the transformer block that uses it. Drop when the upstream PRs land and
``physicsnemo.nn.module.mlp_layers.Mlp`` is imported directly.
"""

import torch
from torch import nn


class Mlp(nn.Module):
    r"""Two-layer MLP.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hidden_features : int, optional, default=None
        Hidden dim; defaults to ``in_features``.
    out_features : int, optional, default=None
        Output dim; defaults to ``in_features``.
    act_layer : type, optional, default=torch.nn.GELU
        Activation layer.
    drop : float, optional, default=0.0
        Dropout rate.

    Forward
    -------
    x : torch.Tensor
        Tokens of shape :math:`(B, N, \text{in\_features})`.

    Outputs
    -------
    torch.Tensor
        Tokens of shape :math:`(B, N, \text{out\_features})`.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
