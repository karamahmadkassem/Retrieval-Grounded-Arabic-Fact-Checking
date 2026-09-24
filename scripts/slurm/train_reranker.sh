#!/bin/bash
#SBATCH --job-name=arafa-rerank
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=32000
#SBATCH --time=0-12:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/reranker-%j.out
#SBATCH --error=logs/reranker-%j.err
#
# Requires data/arafa/localization/biencoder_test.json from the bi-encoder job.
# Edit --account and --gres to match Bassel's onboarding notes.
#   sbatch scripts/slurm/train_reranker.sh
#
# OOM on bge-reranker-v2-m3: use
#   --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --batch-size 16
#
# Breakdown:
#   python scripts/analyze_localization_breakdown.py \
#     --results data/arafa/localization/reranker_test.json

set -euo pipefail
cd "$HOME/Retrieval-Grounded-Arabic-Fact-Checking"
mkdir -p logs models/inarticle_reranker

module purge
module load cuda
module load python/pytorch
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"

test -f data/arafa/localization/biencoder_test.json || {
  echo "Missing data/arafa/localization/biencoder_test.json — run train_biencoder.sh first." >&2
  exit 1
}

python scripts/train_inarticle_reranker.py \
  --model BAAI/bge-reranker-v2-m3 \
  --dense-results data/arafa/localization/biencoder_test.json \
  --epochs 1 \
  --batch-size 8 \
  --eval-split test
