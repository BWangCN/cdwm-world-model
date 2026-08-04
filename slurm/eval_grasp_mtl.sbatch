#!/bin/bash
#SBATCH --job-name=mtl-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:30:00
#SBATCH --nodelist=icing1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== grasp_mtl eval (CPU) | $(hostname) $(date) ==="
python -u eval_grasp_mtl.py --S 8
echo "=== done $(date) ==="
