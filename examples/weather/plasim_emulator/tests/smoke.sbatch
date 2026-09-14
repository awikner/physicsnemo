#!/bin/bash
#SBATCH -J plasim_smoke
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
set -x
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tests/smoke_test.py
