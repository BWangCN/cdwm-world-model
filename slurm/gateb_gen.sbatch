#!/bin/bash
#SBATCH --job-name=gateb-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=2:00:00
#SBATCH --nodelist=icing1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
# Gate B episode generation (CPU MuJoCo) on the CPU-only node icing1. Array over seeds = independent data-gen
# replicas run in parallel (icing1 = 96 cpu). No GPU. Smoke-verified locally (386/400 settled, multimodal).
#   sbatch --array=0-7 slurm/gateb_gen.sbatch      # 8 x 1500 = 12k hammer Gate B episodes
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/gateb
conda activate dynamo-repro
echo "=== gateb-gen seed=$SLURM_ARRAY_TASK_ID | $(hostname) $(date) ==="
python -u generate.py --n 1500 --seed "$SLURM_ARRAY_TASK_ID" --out "hammer_gateb_s${SLURM_ARRAY_TASK_ID}.npz"
echo "=== done seed=$SLURM_ARRAY_TASK_ID $(date) ==="
