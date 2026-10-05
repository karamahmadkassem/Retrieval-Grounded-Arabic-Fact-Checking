#!/bin/bash
#SBATCH --job-name=arafa-bienc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=32000
#SBATCH --time=0-06:00:00
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
# Octopus leaves CUDA_VISIBLE_DEVICES unset on a 2-GPU node. sentence-transformers
# then uses DataParallel and OOMs (job 917001, e5-large, 31.7 GiB full).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

free_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
echo "home_free_kb=${free_kb}"
if [ "${free_kb}" -lt 3000000 ]; then
  echo "Need at least 3GB free in \$HOME before training." >&2
  df -h "$HOME"
  exit 1
fi

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; n=torch.cuda.device_count(); print(torch.cuda.get_device_name(0), 'ngpu', n); assert n==1, 'expected 1 GPU, got %s' % n"

# Smoke already succeeded (job 916998, 500 claims): R@1=0.776, MRR=0.842.
# Do not submit e5-large here: job 917001 OOMed, and full large training
# cannot finish inside the 6-hour gpu partition limit after ~2h of BM25 mining.
#
# This job: e5-small, all 118,664 train claims, full test, one V100, batch 32.
python scripts/train_inarticle_biencoder.py \
  --model intfloat/multilingual-e5-small \
  --epochs 1 \
  --batch-size 32 \
  --n-negatives 7 \
  --eval-split test
