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

"""Multi-model hindcast mixture datapipe for the ai-rossbypalooza recipe.

Loads forecasts from up to ~8 AI weather experts (two zarr archive schemas)
plus IMERG daily-precipitation truth, for training Mixture-of-Weather-Experts
gates on week-2 monsoon rainfall. Submodules are imported directly
(``from datapipes.dataset import HindcastMixtureDataset``); this package
intentionally avoids eager heavy imports.
"""
