#!/bin/bash
#SBATCH -J plasim_noise
#SBATCH -o logs/noise_%j.out
#SBATCH -e logs/noise_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tests/noise_sensitivity.py \
  config/rtx_1gpu.yaml \
  /work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage4_ensemble.pt
