#!/bin/bash
#SBATCH -J plasim_diagA
#SBATCH -o logs/diagA_%j.out
#SBATCH -e logs/diagA_%j.err
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/diagnose_precip.py \
  --config config/rtx_1gpu.yaml \
  --checkpoint /work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage4_ensemble.pt \
  --split test --days 10 --samples 8 \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/diagnose_stage4.json
