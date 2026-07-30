#!/bin/bash
#SBATCH --job-name=vla-demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=0:30:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# short job (per GPU.md: short/interactive -> qos=batch). ManiSkill needs sm_75 -> pin titanrtx (v100 fails).
. ~/.bashrc
SCR=/misc/kcgscratch1/MengyeGroup/yy5259
export MS_ASSET_DIR=$SCR/maniskill_assets TMPDIR=$SCR/tmp XDG_CACHE_HOME=$SCR/.cache HF_HOME=$SCR/hf
# force the real NVIDIA Vulkan ICD (SAPIEN's bundled loader misses it -> intermittent partial renders)
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
  echo "VK ICD forced: $VK_ICD_FILENAMES"
else echo "WARN: no nvidia_icd.json on $(hostname)"; fi
conda activate maniskill
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
python -u render_demo.py
echo "=== done $(date) ==="
