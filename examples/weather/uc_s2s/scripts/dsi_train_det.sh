#!/bin/bash -l

#SBATCH --job-name=uc_s2s_train_det
#SBATCH --output=dsi_%x_%j.out
#SBATCH --error=dsi_%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4,local:disk:2500G
#SBATCH --mail-user=gongbing@uchicago.edu

# ── Local NVME scratch ────────────────────────────────────────────────────────
user="$USER"
id="$SLURM_JOB_ID"
dir="/local/scratch/${user}_${id}"
mkdir -p "${dir}"

echo "Copying h5data to NVME: /net/monsoon/S2S/h5data -> ${dir}/h5data"
cp -r /net/monsoon/S2S/h5data "${dir}/h5data"
echo "Copy complete."

data_dir="${dir}/h5data"
echo "DATA_DIR=${data_dir}"

# ── Config ────────────────────────────────────────────────────────────────────
config_name=exp16_nvidia_v2   # filename (no .yaml) inside conf/

# ── Checkpoint dir on persistent storage ─────────────────────────────────────
ckpt_dir=/net/monsoon/bing/physicsnemo/examples/weather/uc_s2s/results/det
mkdir -p "${ckpt_dir}"
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

/net/monsoon/bing/myenv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --standalone \
    train.py \
    --config-name=${config_name} \
    data_dir=${data_dir} \
    checkpoint_dir=${ckpt_dir} \
    resuming=false
