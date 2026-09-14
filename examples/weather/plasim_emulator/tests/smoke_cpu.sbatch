#!/bin/bash
#SBATCH -J plasim_smoke_cpu
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
set -x
echo "cpu node"
true
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tests/smoke_test.py
