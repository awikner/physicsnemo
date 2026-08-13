#!/bin/bash -l

#SBATCH --job-name=uc_s2s_inference
#SBATCH --output=dsi_%x_%j.out
#SBATCH --error=dsi_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4,local:disk:2500G
#SBATCH --mail-user=gongbing@uchicago.edu

# ── Local NVME scratch (output only) ─────────────────────────────────────────
user="$USER"
id="$SLURM_JOB_ID"
timestamp=$(date +%Y%m%d_%H%M%S)
dir="/local/scratch/${user}_${id}"
mkdir -p "${dir}"

data_dir=/net/monsoon/S2S/h5data
echo "DATA_DIR=${data_dir}"

# ── Persistent output destination ────────────────────────────────────────────
persistent_dir=/net/monsoon/bing/S2S/v2.0/HPC_scripts/results/S2S/${timestamp}
mkdir -p "${persistent_dir}"

# ── Config ────────────────────────────────────────────────────────────────────
config_name=exp16_nvidia_v2   # filename (no .yaml) inside conf/

# ── Output and checkpoint paths ───────────────────────────────────────────────
output_dir=/net/monsoon/bing/physicsnemo/examples/weather/uc_s2s/results/inference_output
ckpt_vae=/net/monsoon/bing/uc_si_checkpoints/checkpoints/c1/ckpt.tar
ckpt_det=/net/monsoon/bing/uc_si_checkpoints/checkpoints/determinstic/ckpt.tar
ckpt_diff=/net/monsoon/bing/uc_si_checkpoints/si_c1_full_state/training_checkpoints/diff_ckpt.tar
ckpt_finetune=/net/monsoon/bing/uc_si_checkpoints/rollout_finetune_finetune_rollout_crps_multipsteps/training_checkpoints/rollout_ckpt_1400.tar

mkdir -p "${output_dir}"
mkdir -p /net/monsoon/bing/physicsnemo/examples/weather/uc_s2s/logs

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "SLURM_CPUS_ON_NODE:    $SLURM_CPUS_ON_NODE"
echo "SLURM_CPUS_PER_TASK:   $SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NTASKS:          $SLURM_NTASKS"
nvidia-smi

# ── Environment ───────────────────────────────────────────────────────────────
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
conda activate /net/monsoon/bing/myenv
export WANDB_MODE=offline

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=5
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# ── Launch ────────────────────────────────────────────────────────────────────
export NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_OF_NODES=${SLURM_JOB_NUM_NODES}  NUM_GPUS=${NUM_GPUS}  JOB_ID=${SLURM_JOB_ID}"

UC_S2S=/net/monsoon/bing/physicsnemo/examples/weather/uc_s2s
cd "${UC_S2S}"

# ── Background rsync: stream NVME output to persistent storage while running ──
echo "Starting background rsync: ${dir} -> ${persistent_dir}"
while true; do
    rsync -a --no-whole-file "${dir}/" "${persistent_dir}/"
    sleep 60
done &
RSYNC_PID=$!

/net/monsoon/bing/myenv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --standalone \
    inference.py \
    --config-name=${config_name} \
    data_dir=${data_dir} \
    output_dir=${output_dir} \
    checkpoint_path_vae_c1=${ckpt_vae} \
    checkpoint_path_det=${ckpt_det} \
    checkpoint_path_diff=${ckpt_diff} \
    checkpoint_path_finetune=${ckpt_finetune} \
    nvme_dir=${dir}

# ── Final sync after inference completes ─────────────────────────────────────
kill "${RSYNC_PID}" 2>/dev/null
rsync -a --no-whole-file "${dir}/" "${persistent_dir}/"
echo "Final sync complete: ${persistent_dir}"
