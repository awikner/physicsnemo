# PlaSim 6-Hourly Weather Emulator (PhysicsNeMo + Makani SFNO)

An autoregressive weather emulator trained on PlaSim `sim52`, stepping every
**6 hours**, with **precipitation as a diagnostic variable** and calibrated
probabilistic output. Intended both as a general-purpose emulator and as a
forecast engine for the AI-RES rare-event-splitting pipeline.

## Design at a glance

```
state      (52, prognostic, fed back)     pl, tas,
                                          ta1-10, ua1-10, va1-10, hus1-10  (sigma)
                                          zg200..zg1000                     (plev)
forcing    ( 8, input only)               lsm, sg, z0, sst, rsdt,
                                          cos_zenith, lat_grid, lon_grid
noise      ( n, input only)               resampled every AR step
diagnostic ( 2, output only, NEVER fed back)  pr_6h, evap
```

Model I/O: **in = 52 + 8 + n_noise**, **out = 52 + 6** (probabilistic: 3
hurdle-Gamma parameters per diagnostic) or **52 + 2** (deterministic baseline).

The 52-channel state deliberately reproduces the SFNO "v11" contract so a trained
checkpoint is a drop-in for AI-RES.

### What this changes relative to v11

| v11 | Problem | Here |
|---|---|---|
| `normalization: zscore` on precip | zero-inflated, heavy-tailed field; L2 collapses to the conditional mean | hurdle-Gamma likelihood |
| `channel_weights: constant` | precip not upweighted | explicit `precip_weight` |
| `multistep_count: 1` | single-step training, AR drift | rollout fine-tune stages |
| deterministic | ensemble spread only from perturbed ICs | per-step noise channels + predictive distribution |

## Data facts (established by inspection; several are traps)

Source: `/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52`

* Grid is **T42 Gaussian, 64 lat x 128 lon** - hence `legendre-gauss` in SFNO, and
  no regridding. Files are literally named `*_gaussian.nc`.
* **1460 timesteps/year (1464 in leap years) = exactly 6-hourly.** Each
  `<year>_gaussian.nc` covers precisely that calendar year, Jan 1 00:00 to
  Dec 31 18:00.
* Leap years follow the **proleptic Gregorian** rule - year 100 is *not* a leap
  year, and the files agree (1460 steps).
* `sigma_data/<year>_gaussian.nc` already contains everything except the ocean
  forcing: `ta/ua/va/hus` on 10 sigma levels, `zg` on 13 pressure levels
  (we take 10), plus `pl/tas/pr_6h/evap/lsm/sg/z0`.
* That NetCDF exists for **years 7-107 only**. Years **108-111 fall back** to
  `h5/sigma_data/<year>_<idx>.h5` (one file per timestep, complete for 7-132).
  Checking `h5/plev_data` instead is misleading - it covers only the years
  *outside* the training range.
* `sst`, `sic`, `rsdt` are **prescribed repeating annual cycles**, verified
  bit-identical between the per-timestep h5 tree and `boundary_data/`, and
  identical across years at the same day-of-year.
* **The leap and non-leap cycles are independently stretched.** You cannot derive
  one from the other by deleting Feb 29 - doing so is wrong by up to 62 K in
  `sst`. There is no non-leap `rsdt` NetCDF, so it is extracted once from the h5
  tree (`--build-rsdt-cache`).

### Channel decisions the data forced

* **`sic` is dropped.** In sim52 it is identically `1.0` wherever defined and NaN
  over land, across all years and timesteps. It carries zero information and its
  zero variance would divide by zero in z-score normalization. (v11 listed it as
  a forcing.)
* **No separate orography channel.** `sg` is `surface_geopotential` (m^2 s^-2),
  i.e. orography x g, and is already a forcing. Adding orography would duplicate it.
* **`evap` is stored non-positive** (`<= 0`, upward-flux convention) and is
  zero-inflated (~10% exact zeros, 20.6% over land). We store `-evap >= 0` and
  model it with the same hurdle-Gamma as precipitation - a Gaussian would put
  mass below zero and smear the point mass at zero.
* **`pr_6h` in the raw files is an accumulation in m/6h** (max ~0.022, i.e.
  ~89 mm/day via `x1000x4`), matching AI-RES `scores/pipeline.py::_extract_pr`.
  The m/s-rate convention applies to the emulator's *output* channel at the
  AI-RES boundary (`PanguPlasimFS.py:204-230`). Do not conflate the two.
* `pl` is **already** `log_surface_pressure` - do not re-log it.
* `sst` is masked over land (~32% of points); the packer fills with the field's
  own valid mean, and `lsm` tells the model where land is.

### Solar geometry is calibrated, not assumed

sim52's subsolar longitude at t=0 is **222.19 deg**, not the 180 deg a naive UTC
convention gives. `cos_zenith` therefore carries a fitted `-42.1875 deg` hour-angle
offset and a 4-day declination phase shift. With those,
`corr(S0*max(cosz,0), rsdt) = 0.990` over daylight. Re-run
`--check-solar` before changing them.

The residual night mismatch is expected: PlaSim's `rsdt` is a **6-hour mean**, so
it stays positive at a sample where the instantaneous sun is just below the horizon.

## Layout

```
channels.py        channel spec - single source of truth
distributions.py   hurdle-Gamma (zero-inflated) distribution
dataset.py         AR sequence dataset over the packed HDF5
model.py           SFNO emulator + the prognostic/diagnostic split; SFNO head
loss.py            lat-weighted state L2; hurdle-Gamma diagnostic NLL
train.py           two-stage curriculum (single-step -> rollout)
config/base.yaml   Model A configuration
tools/pack_plasim.py    raw sim52 -> per-year HDF5
tools/compute_stats.py  normalization stats (TRAIN years only)
tools/submit_pack.sh    SLURM array packer
```

## Usage

```bash
PY=/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python
OUT=/work2/11079/aasch/stampede3/data/plasim_sim52_pnemo

# 0. one-time: non-leap rsdt cycle (no non-leap NetCDF exists)
$PY tools/pack_plasim.py --build-rsdt-cache --out $OUT

# 1. pack (I/O bound - run as a SLURM array, not serially)
sbatch --array=0-9 tools/submit_pack.sh
SPLIT=valid FIRST_YEAR=112 LAST_YEAR=121 sbatch --array=0-4 tools/submit_pack.sh
SPLIT=test  FIRST_YEAR=122 LAST_YEAR=131 sbatch --array=0-4 tools/submit_pack.sh

# 2. stats over TRAIN years only
$PY tools/compute_stats.py --out $OUT

# 3. smoke test, then train
$PY train.py --config config/base.yaml --smoke
srun $PY train.py --config config/base.yaml
```

Splits: **train 12-111** (as specified), **valid 112-121**, **test 122-131**.
Statistics are computed over the training years only.

## Environment

`/work2/11079/aasch/stampede3/pnemo-plasim-venv` - a venv layered on the existing
`makani` conda env (`--system-site-packages`), with our **own** Makani clone
(`/work2/11079/aasch/stampede3/makani-src-aasch`) installed editable.

This matters: the conda env's `makani` is an editable install pointing into
another user's home directory, alongside other paths that are already
permission-denied. The venv shadows it without modifying the shared env, which
AI-RES's forecast path depends on. Setuptools appends its editable finder to
`sys.meta_path`, *after* the normal path finder, so the venv copy wins.

## Measured performance

On **one NVIDIA RTX PRO 6000 Blackwell** (`rtx-small`), 54.5M-parameter SFNO,
batch 32, bf16:

| quantity | value |
|---|---|
| throughput | ~2.7 it/s |
| steps / epoch | 4562 (146k samples / batch 32) |
| epoch time | ~28 min |
| stage1 (50 ep, unroll 1) | ~23 h |
| stage2 (8 ep, unroll 2) | ~8 h |
| stage3 (4 ep, unroll 4) | ~8 h |
| **full curriculum** | **~38 h** |

The default curriculum therefore fits a 48 h wall limit on a single GPU, with
roughly 10 h of margin. Checkpoints are written every epoch, so even a truncated
run yields a usable model plus every completed stage.

Measure throughput from a steady-state window (sample the iteration counter twice
~90 s apart), not from total elapsed time: the latter folds in the preflight,
model construction and dataset index build, which inflated an early estimate here
from 28 to 39 min/epoch and wrongly suggested the curriculum would not fit.

Node inventory found on Stampede3:
* `h100`: **4x NVIDIA H100** per node - but the queue ran ~3.5 days deep.
* `rtx-small`: **2x NVIDIA RTX PRO 6000 Blackwell (96 GB)** per node, usually
  available immediately. One of these fits batch 32 on its own, which reproduces
  v11's global batch without any distributed training.

**Multi-GPU is currently broken on `rtx-small`**: a 2-rank `torch.distributed.run`
launch SIGSEGVs on both ranks *before* the model is constructed (so it is in
distributed init, not the model or the data). Single-GPU on the same node is fine.
`train.py` calls `faulthandler.enable()` so a repeat gives a Python frame rather
than a bare `exitcode: -11`. Until that is diagnosed, use one GPU here, or h100.

## Known gaps

* Number of GPUs per Stampede3 `h100` node is unconfirmed (SLURM reports
  `Gres=(null)`); it sets batch size and the 24 h feasibility estimate.
* v11's run directory and packaged training data are permission-denied, so a
  direct v11-vs-new comparison is limited to the local checkpoint copy at
  `/work2/11079/aasch/stampede3/shared/sfno_weights/v11`.
