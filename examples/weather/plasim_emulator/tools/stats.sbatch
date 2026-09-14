#!/bin/bash
#SBATCH -J plasim_stats
#SBATCH -o logs/stats_%j.out
#SBATCH -e logs/stats_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/compute_stats.py \
  --out /work2/11079/aasch/stampede3/data/plasim_sim52_pnemo --stride 4
