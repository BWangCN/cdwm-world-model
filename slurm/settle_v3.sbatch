#!/bin/bash
#SBATCH --job-name=cdwm-settle-v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=00:40:00
#SBATCH --qos=batch
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%j.out
# Rebuild the H=32 settle target PURELY from gripper_v3 tier A (poses + achieved gripper channels),
# adding grip_target/grip_phase. CPU/numpy only. Writes settle_target.npz into the v3 dir (CDWM_SETTLE).
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro
echo "=== settle-v3 rebuild $(hostname) $(date) ==="
python -u compute_settle_target.py
echo "=== done $(date) ==="
