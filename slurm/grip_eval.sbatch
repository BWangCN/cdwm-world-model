#!/bin/bash
#SBATCH --job-name=grip-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Grip-head metrics on best.pt over full val: MAE + phase-sliced MAE (close/hold) + tolerance accuracy.
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== grip_eval | $(hostname) $(date) ==="; nvidia-smi -L
python -u grip_eval.py
echo "=== done $(date) ==="
