#!/bin/bash
#SBATCH --job-name=gateb-wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=3:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
# Gate B distributional WM, 3-arm array: 0=point 1=diff(no-latent) 2=diff_oracle. DiT diffusion head, V100/32GB.
# The arms ARE the experiment (point vs distribution vs oracle-latent). Smoke-verified (all 3 train+sample on CPU).
#   sbatch --array=0-2 slurm/train_gateb.sbatch
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
ARMS=(point diff diff_oracle); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "=== gateb-wm $ARM | $(hostname) $(date) ==="; nvidia-smi -L
python -u train_gateb.py --arm "$ARM" --tag "$ARM" --epochs 120 --bs 256 --seed 0
echo "=== done $ARM $(date) ==="
