#!/bin/bash
# Train the PlaSim SFNO emulator on one Stampede3 h100 node.
#
#   sbatch tools/submit_train.sh                 # Model A, full curriculum
#   SMOKE=1 sbatch tools/submit_train.sh         # short pipeline check on real data
#
# NOTE on queueing: the h100 partition enforces a per-user concurrent-job cap.
# Submitting an exploratory job while a production run is pending will take the
# slot ahead of it, because pending jobs are served in submit order. Check
# `squeue -u $USER -p h100` before submitting.
#
#SBATCH -J plasim_train_rtx
#SBATCH -o logs/trainrtx_%j.out
#SBATCH -e logs/trainrtx_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00

set -euo pipefail

ROOT=/work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
PY=/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python
CONFIG=${CONFIG:-config/base.yaml}

cd "$ROOT"
mkdir -p logs

echo "=== node/GPU inventory ==="
nvidia-smi -L || true
NGPU=${NGPU_OVERRIDE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
echo "detected $NGPU GPUs"

# DistributedManager.initialize() picks up SLURM_* / torchrun env automatically.
# One process per GPU.
if [ "${SMOKE:-0}" = "1" ]; then
  echo "=== SMOKE RUN (single process) ==="
  $PY train.py --config "$CONFIG" --smoke
  exit $?
fi

# Preflight inside the same allocation: a 24 h slot is expensive and the queue is
# contended, so prove the GPU path works on a tiny model for a few iterations
# before committing to the full curriculum. Costs ~1 minute.
#
# It must use the SAME launcher as the real run. A preflight that runs
# single-process while the real run uses torchrun cannot catch launcher-level
# faults - which is exactly how a wrong-interpreter failure slipped through once.
echo "=== PREFLIGHT (tiny model, few iters, same launcher) ==="
if [ "$NGPU" -gt 1 ]; then
  PREFLIGHT_CMD="$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPU train.py --config $CONFIG --smoke"
else
  PREFLIGHT_CMD="$PY train.py --config $CONFIG --smoke"
fi
if ! $PREFLIGHT_CMD; then
  echo "PREFLIGHT FAILED - not starting the full run" >&2
  exit 1
fi
echo "=== PREFLIGHT OK - starting full run ==="

if [ "$NGPU" -gt 1 ]; then
  # Launch via `$PY -m torch.distributed.run`, never the bare `torchrun` on PATH:
  # TACC's module environment puts /opt/apps/python/3.12/bin/torchrun first, which
  # would spawn workers under the SYSTEM interpreter (no h5py, different torch)
  # while the preflight above silently succeeds because it calls $PY directly.
  $PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    train.py --config "$CONFIG"
else
  $PY train.py --config "$CONFIG"
fi
