#!/bin/bash
#SBATCH -J plasim_r2
#SBATCH -o logs/r2_%j.out
#SBATCH -e logs/r2_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 40:00:00
set -x
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python
CK=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage4_ensemble.pt
CFG=config/rtx_round2.yaml
nvidia-smi -L

# Preflight exercises the NEW code paths: per-stage loss override, the
# reparameterised hurdle-Gamma sampler, and the daily-aggregate CRPS term
# (unroll 8 >= 4 so it activates). Cheap insurance on a 40 h slot.
echo "=== PREFLIGHT ==="
$PY train.py --config $CFG --only-stages r2_unroll8 --resume "$CK" --smoke \
  || { echo "PREFLIGHT FAILED"; exit 1; }
echo "=== PREFLIGHT OK ==="
$PY train.py --config $CFG --resume "$CK"
