<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# Known test failures (2026-08-14, 2026-08-18)

All four are **fixed**. Section 4 was found and fixed on 2026-08-18 and was
never really an upstream bug at all — it was one environment variable the
DeltaAI module leaves set.

Found while running the full suite during the Phase 12 work (2026-08-13);
neither of the first two belonged to that branch (both reproduced with every
Phase-12 change stashed), so they were parked here and **fixed on 2026-08-14**.
A third was found on 2026-08-18 running the suite on Delta and fixed the same
day. Kept as a record: the diagnoses are the reusable part, the first describes a
trap any new recipe directory can fall into, and the third is a trap any
"bit-identical" test can fall into.

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

**Fixed** — `test/recipes/ai_rossbypalooza/conftest.py` gained
`load_recipe_module(name)` plus a `palooza_train` fixture: it pops the
colliding names from `sys.modules`, puts the palooza directory first, imports
fresh, and asserts the loaded module's `__file__` is under that directory.
Popping (rather than aliasing `train` alone) is what makes it correct: **`ema`
collides too**, and palooza's `train` does `from ema import ModelEMA`, so the
whole transitive import has to resolve under the right path. State is restored
afterwards, so either suite order works — verified both ways.

**For any new recipe directory:** the collision recurs for every module name two
recipe dirs share, and it is silent — the wrong module simply lacks the attribute
you wanted, or worse, has one. `comm -12` over the two directory listings is the
quick check:

```bash
comm -12 <(ls examples/weather/ai_rossby/*.py | xargs -n1 basename | sort) \
         <(ls examples/weather/ai_rossbypalooza/*.py | xargs -n1 basename | sort)
```

## 2. ArchesWeather diagnostic-variable guard no longer raises

**Symptom** —
`test/experimental/models/archesweather/test_archesweather.py::test_diagnostic_variables_rejected`

```
with pytest.raises(ValueError, match="no diagnostic head"):
E   Failed: DID NOT RAISE <class 'ValueError'>
```

Failed standalone, so not an ordering artifact.

**Fixed** — the test was stale. Git says it plainly: the port (`d644e77c`,
2026-07-21) was head-less and rejected the argument, the tests (`32aab7d4`, same
day) were written against that, and then `677224c9` (2026-07-27, "Add diagnostic
head (precip + OLR) to ArchesWeather") **gave the model the capability and removed
the guard**. The docstring documents the new behaviour; only the test disagreed.
Replaced with two tests pinning the current contract: a non-empty
`diagnostic_variables` attaches the head and returns a `(b, n_diag, H, W)` tensor,
and the empty default keeps the original geoarches architecture with a scalar 0 in
that slot.


## 3. A "bit-identical" test that was really a thread-count test

**Symptom** — `test/models/amip_si/test_rolling_dit_features.py::test_legacy_
module_tree_and_forward_are_bit_identical` fails on Delta inside an HPC job:

```
AssertionError: legacy forward drifted from the pre-12e reference
```

The printed tensors agree to four decimals, and the same test's *structural*
assertions — state-dict key set, parameter count — pass.

**Not Delta, and not the branch.** It reproduces on a laptop, same commit, same
torch, by changing one environment variable:

| `OMP_NUM_THREADS` | max &#124;diff&#124; | differing elements |
|---|---|---|
| unset / 8 | 0 | 0 of 15360 |
| 2 | 4.77e-7 | 2464 of 15360 |
| 1 | 4.77e-7 | 12907 of 15360 |

4.77e-7 is **one float32 ULP** at this tensor's 1.68 scale. CPU GEMM splits its
reduction across threads, so the summation order — and therefore the last bit —
changes with the thread count. `data/rolling_dit_legacy_v1.pt` was generated
interactively (many threads); the HPC job scripts export `OMP_NUM_THREADS=1`,
which is why this only ever appeared under a scheduler.

A first look pointed at Delta's older torch (2.10 vs 2.11) or its AMD CPUs. Both
were wrong: a Delta **CPU** node with default threading reproduced the reference
*bitwise*. Worth remembering as a debugging lesson — the two loud differences
between the environments were not the cause, and one cheap local experiment
(re-run with the launcher's env var) beat reasoning about the platforms.

**Fixed** — the forward comparison is now `allclose(rtol=0, atol=1e-6)`, ~2x the
ULP and ~6 orders below the signal, with the measurement recorded in the
docstring; the structural half stays exact, since key sets and parameter counts
are integers and strings and catch a moved module immediately. The test is
renamed `..._are_unchanged`, because bitwise was never a property of the code.

**The general trap:** asserting `torch.equal` against a *stored* float tensor
pins the reduction order of whatever machine and thread count generated it.
Bitwise is the right bar for data movement — `test_channel_layouts.py` compares
packing and permutation results with `torch.equal` and is correct to, which is
why those passed here — but not for anything that sums floats.


## 4. 40 `multi_diffusion` compile failures on DeltaAI — `CXX=CC`

**Symptom** — running the suite on a DeltaAI GH200 node, 40 tests fail:

| count | file |
|---|---|
| 22 | `test/diffusion/test_multi_diffusion_losses.py` |
| 16 | `test/diffusion/test_multi_diffusion_predictor.py` |
| 2 | `test/diffusion/test_multi_diffusion_sampling.py` |

All in `TestCompile`-style classes, i.e. `torch.compile` paths. The same tests
**pass on x86_64** (a full local `pytest test/` run the same day: 14371 passed,
0 failed).

**Not the fork's code** — established first with a control run: the untouched
clone at `/work/nvme/bdiu/awikner/physicsnemo` (branch `ai-rossbypalooza`, no RSI
present) failed **exactly 40**, the same three files. Worth doing that control
before attributing a platform failure to your branch; it takes 3.7 minutes.

**Root cause.** Every one of the 40 reports `CppCompileError: C++ compile
error`, and under it:

```
ld: crt1.o: in function `_start': undefined reference to `main'
collect2: error: ld returned 1 exit status
```

which is a link that has lost its `-shared` — an *executable* link of a file
with no `main`. The `python/miniforge3_pytorch/2.10.0` module leaves the Cray
programming environment's wrappers exported as **`CXX=CC`, `CC=cc`**, and
TorchInductor's CPU backend shells out to `$CXX` to build its generated kernels
as a shared object. Through the Cray wrapper that link comes out wrong.

The GCC line printed just above it — `note: parameter passing for argument of
type 'std::pair<...>' when C++17 is enabled changed to match C++14 in GCC 10.1`
— is a *note*, not the error, and the `precompiled_headers/` path in the
traceback is incidental: `TORCHINDUCTOR_CPP_PRECOMPILE_HEADERS=0` does not help.
Both are the kind of loud-but-irrelevant detail that makes this look like a
toolchain incompatibility rather than one stray variable.

**Fixed** — `export CXX=g++ CC=gcc` before pytest (unsetting both also works;
torch then does its own detection). Measured on `gh043`/`gh099`: **40 failed /
550 passed -> 590 passed / 0 failed**, and the run gets *faster* (194 s -> 91 s),
since the doomed compile attempts are no longer retried. Now exported by the
`deltaai-smoke-test` skill and documented in `hpc/deltaai.md`; set it by hand for
ad-hoc `srun` sessions.

**The general trap:** a module that helpfully sets `CC`/`CXX` for MPI builds
silently redirects every other consumer of those variables, including
`torch.compile`. When codegen fails on one cluster and not another, print the
toolchain env before reading the C++.
