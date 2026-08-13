<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# Known test failures (not Phase 12; parked for another branch)

Found while running the full suite during the Phase 12 work (2026-08-13). Both
reproduce with every Phase-12 change stashed, so neither belongs to that branch —
recorded here so they are not rediscovered from scratch.

## 1. Two recipe dirs both export a module named `train`

**Symptom** — 4 tests in `test/recipes/ai_rossbypalooza/test_train_smoke.py`
fail with

```
AttributeError: module 'train' has no attribute 'run'
```

(`test_train_smoke_and_resume`, `test_best_checkpoint_and_early_stopping`,
`test_ema_disabled_path_still_trains`, `test_inference_writes_gate_forecasts`).

**Only under a combined run.** `pytest test/recipes/ai_rossbypalooza` alone is
168 passed; `pytest test/recipes` is 4 failed. Minimal reproducer:

```bash
pytest test/recipes/ai_rossby test/recipes/ai_rossbypalooza/test_train_smoke.py -q
```

**Cause** — both suites do `sys.path.insert(0, <their recipe dir>)` and then
`import train`, but `examples/weather/ai_rossby/train.py` and
`examples/weather/ai_rossbypalooza/train.py` are *different modules with the same
name*. Whichever suite imports first owns `sys.modules["train"]` for the rest of
the session, so the palooza tests end up calling into the ai_rossby recipe, which
has no `run()`.

**Fix directions** — load each recipe module under a distinct name via
`importlib.util.spec_from_file_location` (e.g. `palooza_train`), or add a
palooza-side fixture that pops the shared names (`train`, `train_loop`,
`dataset_setup`, `inference`, …) from `sys.modules` around the import. The first
is sturdier; the collision will otherwise recur for every module name the two
recipes share.

## 2. ArchesWeather diagnostic-variable guard no longer raises

**Symptom** —
`test/experimental/models/archesweather/test_archesweather.py::test_diagnostic_variables_rejected`

```
with pytest.raises(ValueError, match="no diagnostic head"):
E   Failed: DID NOT RAISE <class 'ValueError'>
```

Fails standalone, so it is not an ordering artifact. Either the constructor guard
was dropped/renamed while the model gained a diagnostic path, or the test is
asserting an intent the model no longer has — resolve by deciding which is
current, not by relaxing the match string.
