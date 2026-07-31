#!/bin/bash
#SBATCH --job-name=grasp-eval-suite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Expanded grasp-WM eval suite: point accuracy (rot+trans ADE/FDE, ADD/ADD-S) + distribution (miss-rate/coverage/
# spread) + physical plausibility (orthonormality/OOD-vs-p99/residual-motion) + slip-gate (AUROC/ECE).
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== grasp_eval_suite | $(hostname) $(date) ==="; nvidia-smi -L
python -u grasp_eval_suite.py --S 8 --rot_tol_deg 10 --pos_tol_frac 0.10
echo "=== done $(date) ==="
