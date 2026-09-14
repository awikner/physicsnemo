#!/bin/bash
#SBATCH -J plasim_smokedata
#SBATCH -o logs/smokedata_%j.out
#SBATCH -e logs/smokedata_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python train.py --config config/base.yaml --smoke
