#!/bin/bash
# Pack PlaSim sim52 into the PhysicsNeMo training layout.
#
# Packing is I/O bound on Lustre (a single year runs at <5% CPU), so the win comes
# from running many years concurrently rather than from more cores per year. Each
# array task takes a block of years and runs them as concurrent processes on one
# node.
#
#   sbatch --array=0-9 submit_pack.sh          # years 12-111 -> train
#
#SBATCH -J plasim_pack
#SBATCH -o logs/pack_%A_%a.out
#SBATCH -e logs/pack_%A_%a.err
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -euo pipefail

ROOT=/work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
OUT=/work2/11079/aasch/stampede3/data/plasim_sim52_pnemo
PY=/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python

# Year blocks. Train 12-111; valid 112-121; test 122-131.
FIRST_YEAR=${FIRST_YEAR:-12}
LAST_YEAR=${LAST_YEAR:-111}
SPLIT=${SPLIT:-train}
NTASKS=${NTASKS:-10}
CONCURRENCY=${CONCURRENCY:-10}

TASK=${SLURM_ARRAY_TASK_ID:-0}

cd "$ROOT"
mkdir -p logs

# Deal out years round-robin so each task gets a similar mix of leap/non-leap.
YEARS=()
for ((y = FIRST_YEAR; y <= LAST_YEAR; y++)); do
  if (( (y - FIRST_YEAR) % NTASKS == TASK )); then
    YEARS+=("$y")
  fi
done

echo "task $TASK packing ${#YEARS[@]} years into $SPLIT: ${YEARS[*]}"

printf '%s\n' "${YEARS[@]}" | xargs -P "$CONCURRENCY" -I{} \
  "$PY" tools/pack_plasim.py --years {} --split "$SPLIT" --out "$OUT"

echo "task $TASK done"
