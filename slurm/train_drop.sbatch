#!/bin/bash
#SBATCH --job-name=drop-wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48GB
#SBATCH --time=4:00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --output=/misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model/slurm_outputs/%x_%A_%a.out
# Baseline drop WM. Array of 3 arms (each varies ONE thing -> no confound):
#   0 full/release  = cloud in release frame + release params  (the baseline)
#   1 pose_only/release = cloud zeroed, release params only     (control: is the cloud needed?)
#   2 full/obj      = cloud in object-canonical frame           (frame ablation: does release-frame help?)
# ~5M-param encoder, tiny -> any GPU; real cost is the CPU dataloader (12 CPUs). Smoke-verified on CPU.
#   sbatch --array=0-2 slurm/train_drop.sbatch
. ~/.bashrc
cd /misc/kcgscratch1/MengyeGroup/yy5259/projects/crus/cdwm-world-model
conda activate dynamo-repro

CFG=("full release" "pose_only release" "full obj")
read MOD FRAME <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
TAG="${MOD}_${FRAME}"
echo "=== drop-wm $TAG (mod=$MOD frame=$FRAME) | $(hostname) $(date) ==="
nvidia-smi -L
CDWM_MODALITY=$MOD python train_drop.py --tag "$TAG" --frame "$FRAME" --epochs 40 --bs 256 --seed 0
echo "=== done $TAG $(date) ==="
