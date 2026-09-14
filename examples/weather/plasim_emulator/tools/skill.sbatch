#!/bin/bash
#SBATCH -J plasim_skill
#SBATCH -o logs/skill_%j.out
#SBATCH -e logs/skill_%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
cd /work2/11079/aasch/stampede3/physicsnemo/examples/weather/plasim_emulator
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/work2/11079/aasch/stampede3/pnemo-plasim-venv/bin/python tools/compare_skill.py \
  --config config/rtx_1gpu.yaml --split test --days 15 --samples 12 \
  --regions Chicago,Bay_of_Bengal \
  --models "v11=builtin,stage1=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage1_single.pt,stage2=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage2_unroll2.pt,stage3=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage3_unroll4.pt,stage4=/work2/11079/aasch/stampede3/runs/plasim_pnemo/modelA_rtx1/ckpt_stage4_ensemble.pt" \
  --out /work2/11079/aasch/stampede3/runs/plasim_pnemo/skill_comparison.json
