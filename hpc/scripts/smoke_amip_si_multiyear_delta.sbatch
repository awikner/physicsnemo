#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Delta twin of smoke_amip_si_multiyear_polaris.pbs (2026-08-18).
#
# Written because Polaris went down with the AMIP SI training verification still
# outstanding. Same four stages and the same assertions; the differences are all
# environmental, and each one is a trap worth naming:
#
#   * DATA. Delta held NONE of the daily-average AMIP stores — its `amip` store
#     is 1981-only, full-res, and its names predate the contract
#     (`sea_surface_temperature`, not `..._monthly_interp`; `specific_total_water`,
#     not `specific_humidity`). The stores are staged from Stampede3 by Globus;
#     $AI_ROSSBY_DATA below points at wherever that landed.
#   * NOISE-SCALE YEARS. The Polaris job hardcodes 1979-2016 because the whole
#     archive is on Eagle. Here the range is DERIVED from what is actually staged,
#     so a partial stage produces a correct artifact for the years it has rather
#     than dying on a missing year. The sidecar .json records the range, which is
#     what keeps a subset artifact honest.
#   * VENV. physicsnemo is installed EDITABLE in the shared venv, pointing at the
#     main repo checkout (a different branch). PYTHONPATH shadows it — verified
#     on this venv, not assumed, because an editable install's meta-path finder
#     can win over sys.path and would silently run the other branch's code.
#   * /work/hdd/bdiu is over its soft quota (20.79T of 19.53T, 21.48T hard), so
#     staging goes to /work/nvme/bdiu unless overridden.
#
# Stage -1 runs the test subset on the compute node before any of it: the login
# nodes are the wrong place to import physicsnemo (CUDA/Warp init), and a green
# suite here is what says the failure below is about data or GPUs, not the env.
#
# Submit:  sbatch hpc/scripts/smoke_amip_si_multiyear_delta.sbatch

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100g
#SBATCH --job-name=amip-si-multiyear
#SBATCH --output=hpc/scripts/logs/amip-si-multiyear-%j.out

# Stages record status and the job fails at the END, so one failure does not hide
# the other results (the lesson from the Polaris verification runs).
set -uo pipefail
FAILED=()

REPO=/work/nvme/bdiu/awikner/physicsnemo-amip-v2
VENV=/work/nvme/bdiu/awikner/physicsnemo/.venv     # shared; see the PYTHONPATH note above
PY=$VENV/bin/python
# Training goes through torchrun, NOT bare python (measured 2026-08-18). An
# sbatch script's shell has SLURM_PROCID=0 and SLURM_NPROCS=1 but NO
# SLURM_LAUNCH_NODE_IPADDR (only srun steps get that). DistributedManager tries
# initialize_env() first and falls through to the SLURM branch on TypeError,
# where it reads that missing variable as its `addr` and hands None to setup().
# torchrun --standalone sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT, so the ENV
# path wins and the launch is deterministic — the same reason every other Delta
# script here uses torchrun. The noise-scale builder stays bare python: it never
# touches DistributedManager.
TORCHRUN="$VENV/bin/torchrun --standalone --nproc-per-node=1"
cd "$REPO"

export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AI_ROSSBY_DATA=${AI_ROSSBY_DATA:-/work/nvme/bdiu/awikner/physicsnemo-zarr}

STATE=$AI_ROSSBY_DATA/amip_dailyavg_coarse
BND=$AI_ROSSBY_DATA/amip_dailyavg_boundary
SIGMA=$AI_ROSSBY_DATA/norm_stats/sigma_c_amip_dailyavg_coarse.pt
RUN=/work/nvme/bdiu/awikner/si-multiyear-smoke
CKPT_DIR=$RUN/checkpoints
TSV=$RUN/per_batch.tsv
LOG=$RUN/train.log
RESUME_LOG=$RUN/resume.log
mkdir -p "$RUN" "$(dirname "$SIGMA")" hpc/scripts/logs

# The editable install points elsewhere; prove the shadow took before anything
# else runs, or every result below is about the wrong checkout.
echo "############################################################"
echo "### -1: environment"
echo "############################################################"
echo "repo:      $(git log --oneline -1 | cat)"
echo "python:    $($PY -V 2>&1)"
RESOLVED=$($PY -c "import physicsnemo; print(physicsnemo.__file__)")
echo "physicsnemo -> $RESOLVED"
case "$RESOLVED" in
    "$REPO"/*) echo "PYTHONPATH shadow: OK" ;;
    *) echo "ERROR: physicsnemo resolves OUTSIDE this worktree — the editable"
       echo "       install won. Everything below would test the other branch."
       exit 1 ;;
esac
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
       torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
$PY -c "import muon" 2>/dev/null && echo "optimizer: Muon available" || echo "optimizer: muon MISSING"
echo

echo "--- test subset ---"
if $PY -m pytest test/recipes/ai_rossby test/datapipes/climate test/tools/data \
        test/models/amip_si -q 2>&1 | tail -3
then echo "STAGE -1: PASS"; else echo "STAGE -1: FAIL"; FAILED+=("-1:pytest"); fi
echo

# ---------------------------------------------------------------------------
# Stage 0 — pre-flight: the two archives must agree on years
# ---------------------------------------------------------------------------
echo "############################################################"
echo "### 0: pre-flight — staged archives and their year sets"
echo "############################################################"
for d in "$STATE" "$BND"; do
    [[ -d "$d" ]] || { echo "ERROR: $d missing — stage the Globus transfer first"; exit 1; }
done
( cd "$STATE" && ls -d *.zarr 2>/dev/null | sort ) > "$RUN/state_years.txt"
( cd "$BND"   && ls -d *.zarr 2>/dev/null | sort ) > "$RUN/bnd_years.txt"
NSTATE=$(wc -l < "$RUN/state_years.txt")
echo "state:    $NSTATE store(s)  [$(head -1 "$RUN/state_years.txt") .. $(tail -1 "$RUN/state_years.txt")]"
echo "boundary: $(wc -l < "$RUN/bnd_years.txt") store(s)  [$(head -1 "$RUN/bnd_years.txt") .. $(tail -1 "$RUN/bnd_years.txt")]"
(( NSTATE >= 2 )) || { echo "ERROR: need >= 2 year stores to exercise the multi-year path"; exit 1; }
# A state year with no boundary year now RAISES in the loader (a missing year
# would otherwise be served another year's SST, silently — boundary reads are by
# day-of-year, so any store answers). Catch it here, not 30 GPU-minutes in.
ONLY_STATE=$(comm -23 "$RUN/state_years.txt" "$RUN/bnd_years.txt" | tr '\n' ' ')
if [[ -n "${ONLY_STATE// }" ]]; then
    echo "ERROR: state years with no boundary store: $ONLY_STATE"
    exit 1
fi
for f in "$STATE/normalize_mean_dailyavg.nc" "$STATE/normalize_std_dailyavg.nc"; do
    [[ -f "$f" ]] || { echo "ERROR: $f missing (norm stats live at the store root)"; exit 1; }
done
# Derived, not hardcoded — this archive may be a subset.
Y0=$(head -1 "$RUN/state_years.txt" | sed 's/\.zarr//')
Y1=$(( $(tail -1 "$RUN/state_years.txt" | sed 's/\.zarr//') + 1 ))
echo "year sets match; noise scales will cover $Y0-$((Y1 - 1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ---------------------------------------------------------------------------
# Stage 1 — per-channel noise scales from the staged years
# ---------------------------------------------------------------------------
echo "############################################################"
echo "### 1: noise scales — $Y0-$((Y1 - 1)), amip_si pack order"
echo "############################################################"
# --sample-stride 4: one increment per day rather than per row. The std is
# converged long before 6-hourly rows are exhausted and the builder is I/O bound.
if $PY tools/data/amip/make_noise_scales.py \
    --zarr "$STATE" \
    --model-config examples/weather/ai_rossby/conf/model/amip_si.yaml \
    --mean "$STATE/normalize_mean_dailyavg.nc" \
    --std  "$STATE/normalize_std_dailyavg.nc" \
    --year-start "$Y0" --year-end "$Y1" \
    --sample-stride 4 \
    --out "$SIGMA"
then echo "STAGE 1: PASS"; else echo "STAGE 1: FAIL"; FAILED+=("1:noise-scales"); fi
echo "--- sidecar (records the year range, which is what keeps a subset honest) ---"
head -30 "${SIGMA%.pt}.json" 2>/dev/null || echo "(none)"
echo

# ---------------------------------------------------------------------------
# Stage 2 — 250 iterations over the multi-year archive
# ---------------------------------------------------------------------------
echo "############################################################"
echo "### 2: train_diffusion — model=amip_si loss=si multi-year"
echo "############################################################"
NOISE_OVERRIDE=()
if [[ -f "$SIGMA" ]]; then
    NOISE_OVERRIDE=(++loss.noise_scale_path="$SIGMA")
else
    echo "NOTE: no sigma artifact — training with isotropic noise instead"
fi
rm -rf "$TSV" "$LOG" "$CKPT_DIR"
# hydra job.chdir=True moves the cwd, so the TSV path is absolute on purpose.
if $TORCHRUN examples/weather/ai_rossby/train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_dailyavg_coarse_multiyear \
    training=amip_diffusion \
    validation=off \
    "${NOISE_OVERRIDE[@]}" \
    ++training.max_epochs=1 \
    ++training.stages.0.num_epochs=1 \
    ++training.stages.0.max_iterations=250 \
    ++training.ema.warmup_epochs=0 \
    ++bench.per_batch_tsv="$TSV" \
    +checkpoint_dir="$CKPT_DIR" \
    run_name=si_multiyear_smoke_delta \
    wandb.enabled=False 2>&1 | tee "$LOG"
then echo "STAGE 2 (run): PASS"; else echo "STAGE 2 (run): FAIL"; FAILED+=("2:train"); fi
echo

echo "--- multi-year routing + stride ---"
grep -E "multi-year archive|model step:" "$LOG" || echo "(neither line present)"
if grep -q "model step: 4 store row(s) (24 h)" "$LOG"; then
    echo "ASSERT stride: PASS"
else
    echo "ASSERT stride: FAIL — a 24 h model over 6 h rows must step 4 rows"
    FAILED+=("2:stride")
fi
if grep -q "multi-year archive" "$LOG"; then
    echo "ASSERT multi-year routing: PASS"
else
    echo "ASSERT multi-year routing: FAIL — a directory did not route, so this"
    echo "  trained one year"
    FAILED+=("2:routing")
fi

echo "--- loss health ---"
# NOT "the loss must fall". CONVERGENCE.md records a HEALTHY run of this family
# moving -4.6% over 7300 batches, with per-batch variance far larger than that;
# 250 iterations cannot resolve a trend and demanding one fails good runs.
if $PY - "$TSV" <<'PY'
import math, pathlib, statistics, sys
rows = [l.split("\t") for l in pathlib.Path(sys.argv[1]).read_text().splitlines()[1:] if l.strip()]
loss = [float(r[3]) for r in rows]
if len(loss) < 40:
    raise SystemExit(f"only {len(loss)} iterations logged")
bad = [i for i, v in enumerate(loss) if not math.isfinite(v)]
if bad:
    raise SystemExit(f"non-finite loss at iterations {bad[:10]}")
B = 40
for i in range(0, len(loss) // B * B, B):
    blk = loss[i:i + B]
    print(f"  iters {i:>4}-{i + B - 1:<4} mean {statistics.mean(blk):>12.4e}  "
          f"sd {statistics.stdev(blk):>12.4e}")
first, last = statistics.mean(loss[:B]), statistics.mean(loss[-B:])
sd = statistics.stdev(loss[:B])
print(f"iterations {len(loss)}  first{B} {first:.4e} -> last{B} {last:.4e}  "
      f"({100 * (last / first - 1):+.1f}%, block-mean noise ~{100 * sd / math.sqrt(B) / first:.1f}%)")
if last > 1.5 * first:
    raise SystemExit("loss DIVERGED (>1.5x its opening block)")
print("ASSERT loss finite and not diverging: PASS")
PY
then :; else echo "ASSERT loss health: FAIL"; FAILED+=("2:loss"); fi
echo

# ---------------------------------------------------------------------------
# Stage 3 — the checkpoint round-trips
# ---------------------------------------------------------------------------
# Trainable has to include resumable: no 37-year run finishes in one allocation,
# and this partition caps at 50 minutes. Re-entering the same checkpoint_dir puts
# the fresh artifact through assert_checkpoint_dir_contract, which compares the
# .mdlus's stored channel contract to the live model — every contract bug in the
# Phase 12 rebaseline was shape-preserving, so "it loaded" proves nothing.
echo "############################################################"
echo "### 3: resume — contract guard + load of our own checkpoint"
echo "############################################################"
if $TORCHRUN examples/weather/ai_rossby/train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_dailyavg_coarse_multiyear \
    training=amip_diffusion \
    validation=off \
    "${NOISE_OVERRIDE[@]}" \
    ++training.max_epochs=2 \
    ++training.stages.0.num_epochs=2 \
    ++training.stages.0.max_iterations=20 \
    ++training.ema.warmup_epochs=0 \
    +checkpoint_dir="$CKPT_DIR" \
    run_name=si_multiyear_smoke_delta \
    wandb.enabled=False 2>&1 | tee "$RESUME_LOG"
then echo "STAGE 3 (run): PASS"; else echo "STAGE 3 (run): FAIL"; FAILED+=("3:resume"); fi
# The strings a resume actually emits, confirmed by running one: nothing in the
# log contains the word "resume".
grep -E "channel contract verified|Loaded (model state|checkpoint file|optimizer)|epoch [0-9]+ batch 0/" \
    "$RESUME_LOG" | head -12
if grep -q "channel contract verified against" "$RESUME_LOG"; then
    echo "ASSERT contract guard ran on our own artifact: PASS"
else
    echo "ASSERT contract guard: FAIL — no .mdlus found, so stage 2 wrote no"
    echo "  checkpoint and the restart path is unproven"
    FAILED+=("3:no-guard")
fi
if grep -q "epoch 2 batch" "$RESUME_LOG"; then
    echo "ASSERT resumed epoch numbering: PASS"
else
    echo "ASSERT resumed epoch numbering: FAIL — restarted at epoch 1"
    FAILED+=("3:no-resume")
fi
echo

echo "############################################################"
if ((${#FAILED[@]})); then
    echo "### SI multi-year smoke (Delta): ${#FAILED[@]} check(s) FAILED: ${FAILED[*]}"
    exit 1
fi
echo "### SI multi-year smoke (Delta): all checks PASSED"
