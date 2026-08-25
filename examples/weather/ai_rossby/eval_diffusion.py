# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alias for :mod:`climate_eval_suite` — kept so existing launchers, docs and
imports keep working.

The suite it used to define (five validator subclasses, one full rollout
EACH) was fused into scorer plug-ins riding a single
:class:`~validate_diffusion.DiffusionRolloutValidator` rollout, and the
driver became generation-agnostic (deterministic train.py checkpoints run
through :mod:`deterministic_adapter`). Everything lives in
``climate_eval_suite.py`` now; invoke either module — the Hydra entrypoint
and config contract (``validation=eval_suite``) are identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from climate_eval_suite import (  # noqa: F401,E402
    ClimatologyScorer,
    EvalSuiteRunner,
    FluxSeriesScorer,
    QBOScorer,
    VariableCatalog,
    _estimate_period_months,
    _perturber_scales,
    _resolve_eval_sampler_num_steps,
    _to_cpu,
    _tropical_band_mask_and_weights,
    check_deterministic_ensemble,
    cli,
    derive_spread_skill,
    main,
    resolve_steps_per_bin,
    scan_rmse_trace,
)

__all__ = [
    "ClimatologyScorer",
    "QBOScorer",
    "FluxSeriesScorer",
    "EvalSuiteRunner",
    "VariableCatalog",
    "derive_spread_skill",
    "scan_rmse_trace",
    "resolve_steps_per_bin",
    "check_deterministic_ensemble",
    "main",
    "cli",
]


if __name__ == "__main__":
    cli()
