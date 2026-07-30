#!/bin/bash
#SBATCH --job-name=mjrender
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:20:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
. ~/.bashrc
conda activate maniskill
export MUJOCO_GL=egl
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
for O in 006_mustard_bottle 021_bleach_cleanser 022_windex_bottle 048_hammer 029_plate; do
  echo "--- rendering $O ---"
  python -u render_mujoco.py $O
done
echo "=== done $(date) ==="
