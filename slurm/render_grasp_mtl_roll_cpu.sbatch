#!/bin/bash
#SBATCH --job-name=grasp-mtl-roll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --nodelist=icing1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Render the grasp-MTL rollout (new grip head): 1 rigid held-out episode, N=8 samples, 50 DDIM steps, jaws driven by the
# model's PREDICTED gripper closure. CPU (avoids the GPU queue). -> figures/rollout_grasp_mtl.{mp4,gif}
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== grasp-mtl rollout render (CPU) | $(hostname) $(date) ==="
python -u render_grasp_mtl_roll.py
echo "=== done $(date) ==="
