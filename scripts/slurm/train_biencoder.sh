#!/bin/bash
#SBATCH --job-name=arafa-bienc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=32000
#SBATCH --time=0-12:00:00
#SBATCH --account=kma88
#SBATCH --output=logs/biencoder-%j.out
#SBATCH --error=logs/biencoder-%j.err
#
# SSH: kma88@octopus.aub.edu.lb  (email is kma88@mail.aub.edu).
# If sbatch rejects --account=kma88, replace with the project name Bassel gives you.
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/slurm/train_biencoder.sh
#   squeue -u "$USER"
#   tail -f logs/biencoder-<jobid>.out
#
# Success bar (test): beat in-article BM25 R@1=0.716, MRR=0.801
# Outputs: models/inarticle_biencoder/
#          data/arafa/localization/biencoder_test.json

set -euo pipefail
cd "$HOME/Retrieval-Grounded-Arabic-Fact-Checking"
mkdir -p logs models/inarticle_biencoder

module purge
module load cuda
# python/pytorch is Python 3.6 and cannot run this repo. Use ai-4 (3.10 + torch 2.1).
module load python/ai-4
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"

# GPU smoke (uncomment first; minutes–tens of minutes):
# python scripts/train_inarticle_biencoder.py \
#   --model intfloat/multilingual-e5-small \
#   --epochs 1 --max-train-claims 4000 --batch-size 16 --max-eval-claims 500 \
#   --n-negatives 7 --eval-split test

# Full train. OOM? switch model to intfloat/multilingual-e5-small and/or batch-size 8.
python scripts/train_inarticle_biencoder.py \
  --model intfloat/multilingual-e5-large \
  --epochs 1 \
  --batch-size 16 \
  --n-negatives 7 \
  --eval-split test
