#!/bin/bash
#SBATCH --job-name=grasp-mtl-roll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=0:40:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# MuJoCo replay of the grasp-MTL rollout (real 2F-85 + real object mesh, GT vs predicted). Needs a GPU for EGL offscreen
# render + model sampling. Pass ARG=--smoke for the k=0/k=last placement check. -> figures/rollout_grasp_mtl.{mp4,gif}
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
export MUJOCO_GL=egl
echo "=== grasp-mtl mujoco render ${ARG} | $(hostname) $(date) ==="; nvidia-smi -L
python -u render_grasp_mtl_roll.py ${ARG}
echo "=== done $(date) ==="
