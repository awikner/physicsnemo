# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""ArchesWeather deterministic 3D Swin U-Net weather model.

Port of the ArchesWeather-M backbone from INRIA/geoarches (BSD-3-Clause),
adapted to the ai-rossby PLASIM/ERA5 variable-routing convention and our
17-level / 1-degree ERA5 grid. See :class:`ArchesWeather` for full docs.
"""

from .archesweather import ArchesWeather

__all__ = ["ArchesWeather"]
