#!/bin/bash
#SBATCH -J plasim_eval_amd
#SBATCH -o logs/evalamd_%j.out
#SBATCH -e logs/evalamd_%j.err
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
set -x
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python evaluate.py \
  --config config/rtx_1gpu.yaml \
  --checkpoint /work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage2_unroll2.pt \
  --split test --lead-steps 40 --members 16 --samples 16 \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/eval_stage2_amd.json
