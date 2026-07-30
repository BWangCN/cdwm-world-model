#!/bin/bash
#SBATCH --job-name=vla-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=0:30:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:titanrtx:1
# dry corpus across several known objects (GPU.md: short -> qos=batch; ManiSkill needs sm_75 -> titanrtx)
#SBATCH --array=0-4
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
. ~/.bashrc
SCR=/misc/kcgscratch1/MengyeGroup/yy5259
export MS_ASSET_DIR=$SCR/maniskill_assets TMPDIR=$SCR/tmp XDG_CACHE_HOME=$SCR/.cache HF_HOME=$SCR/hf
[ -f /etc/vulkan/icd.d/nvidia_icd.json ] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
conda activate maniskill
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
OBJS=(006_mustard_bottle 021_bleach_cleanser 022_windex_bottle 048_hammer 029_plate)
O=${OBJS[$SLURM_ARRAY_TASK_ID]}
echo "=== corpus episode: $O ==="
export CDWM_OBJ=$O
python -u render_demo.py
python -u pi0_writer.py figures/vla_trajectory_${O}.npz figures/vla_records/episode_${O}
echo "=== done $O $(date) ==="
