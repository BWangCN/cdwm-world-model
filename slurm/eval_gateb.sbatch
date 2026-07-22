#!/bin/bash
#SBATCH --job-name=gateb-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=1:30:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Gate B payoff eval (diffusion sampling on GPU): point vs diff vs diff_oracle -> P(basin) NLL/entropy. Needs the
# 3 trained arms (gateb_runs/{point,diff,diff_oracle}/best.pt). Smoke-verified (1 batch, all arms, basin assign OK).
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== gateb-eval | $(hostname) $(date) ==="
python -u eval_gateb.py
echo "=== done $(date) ==="
