#!/bin/bash
#SBATCH -J plasim_diagR2
#SBATCH -o logs/diagR2_%j.out
#SBATCH -e logs/diagR2_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=48
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/diagnose_precip.py \
  --config config/rtx_1gpu.yaml \
  --checkpoint /work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_r2/ckpt_r2_unroll32.pt \
  --split test --days 10 --samples 8 \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/diagnose_r2_unroll32.json
