#!/bin/bash
#SBATCH -J plasim_spread
#SBATCH -o logs/spread_%j.out
#SBATCH -e logs/spread_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export OMP_NUM_THREADS=48
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/diagnose_spread.py \
  --config config/rtx_1gpu.yaml --days 10 --samples 6 \
  --models "stage4=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage4_ensemble.pt,r2=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_r2/ckpt_r2_unroll32.pt" \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/spread_diag.json
