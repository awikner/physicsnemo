#!/bin/bash
#SBATCH -J plasim_stage4
#SBATCH -o logs/stage4_%j.out
#SBATCH -e logs/stage4_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
set -x
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python
CK=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage3_unroll4.pt

echo "=== PREFLIGHT: exercise the ensemble/energy-score path ==="
$PY train.py --config config/rtx_1gpu.yaml --only-stages stage4_ensemble \
  --resume "$CK" --smoke || { echo "PREFLIGHT FAILED"; exit 1; }
echo "=== PREFLIGHT OK ==="
$PY train.py --config config/rtx_1gpu.yaml --only-stages stage4_ensemble --resume "$CK"
