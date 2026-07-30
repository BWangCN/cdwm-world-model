#!/bin/bash
#SBATCH --job-name=vla-grasp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=0:20:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
# short job (GPU.md: short->qos=batch); ManiSkill needs sm_75 -> titanrtx
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
. ~/.bashrc
SCR=/misc/kcgscratch1/MengyeGroup/yy5259
export MS_ASSET_DIR=$SCR/maniskill_assets TMPDIR=$SCR/tmp XDG_CACHE_HOME=$SCR/.cache HF_HOME=$SCR/hf
conda activate maniskill
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
python -u grasp_sampler.py
echo "=== done $(date) ==="
