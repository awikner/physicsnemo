#!/bin/bash
#SBATCH -J plasim_skillr2
#SBATCH -o logs/skillr2_%j.out
#SBATCH -e logs/skillr2_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export OMP_NUM_THREADS=48
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/compare_skill.py \
  --config config/rtx_1gpu.yaml --split test --days 15 --samples 12 \
  --regions Chicago,Bay_of_Bengal \
  --models "r2=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_r2/ckpt_r2_unroll32.pt" \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/skill_r2_only.json
