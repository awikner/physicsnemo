#!/bin/bash
#SBATCH -J plasim_stage3
#SBATCH -o logs/stage3_%j.out
#SBATCH -e logs/stage3_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
set -x
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python train.py \
  --config config/rtx_1gpu.yaml \
  --resume /work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage2_unroll2.pt \
  --only-stages stage3_unroll4
