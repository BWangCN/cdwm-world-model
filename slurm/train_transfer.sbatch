#!/bin/bash
#SBATCH --job-name=xfer-wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=4:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
# Cross-object latent transfer: 4 arms on the OBJECT-DISJOINT split (train_gateb sets CDWM_GATEB_LATENT per arm).
#   sbatch --array=0-3 slurm/train_transfer.sbatch
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
ARMS=(no_latent abstract_oracle grounded_oracle shuffled_oracle); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "=== xfer $ARM (object-disjoint) | $(hostname) $(date) ==="; nvidia-smi -L
CDWM_GATEB_SPLIT=object python -u train_gateb.py --arm "$ARM" --epochs 120 --bs 256 --seed 0
echo "=== done $ARM $(date) ==="
