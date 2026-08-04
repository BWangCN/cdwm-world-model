#!/bin/bash
#SBATCH --job-name=grasp-mtl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=16:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Clean multi-task grasp WM, from scratch (shared trunk + private branches; trajectory masked to rigid + boolean on all).
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== grasp_mtl | $(hostname) $(date) ==="; nvidia-smi -L
python -u train_grasp_mtl.py --tag grasp_mtl --epochs 200 --bs 128
echo "=== done $(date) ==="
