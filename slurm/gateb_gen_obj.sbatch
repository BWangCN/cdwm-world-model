#!/bin/bash
#SBATCH --job-name=gateb-obj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=3:00:00
#SBATCH --nodelist=icing1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
# Gate B geometry-consistent multi-object generation (CPU MuJoCo, point-cloud hulls) on icing1. Array over the
# CoM-sensitive object list (gateb/objlist.txt), one object per task. Smoke-verified (clamp 69% settle, multimodal).
#   sbatch --array=0-9 slurm/gateb_gen_obj.sbatch
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/gateb
conda activate dynamo-repro
OBJ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${OBJLIST:-objlist.txt}")
echo "=== gateb-obj $OBJ | $(hostname) $(date) ==="
python -u generate_obj.py --obj "$OBJ" --n 2000 --seed 0
echo "=== done $OBJ $(date) ==="
