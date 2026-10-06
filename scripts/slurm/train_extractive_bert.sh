#!/bin/bash
#SBATCH --job-name=arafa-bert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=32000
#SBATCH --time=0-06:00:00
#SBATCH --account=kma88
#SBATCH --output=logs/extractive-bert-%j.out
#SBATCH --error=logs/extractive-bert-%j.err
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/slurm/train_extractive_bert.sh

set -euo pipefail
cd "$HOME/Retrieval-Grounded-Arabic-Fact-Checking"
mkdir -p logs models/extractive_bert_v3

module purge
module load cuda
module load python/ai-4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

free_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
echo "home_free_kb=${free_kb}"
if [ "${free_kb}" -lt 3000000 ]; then
  echo "Need at least 3GB free in \$HOME before training." >&2
  df -h "$HOME"
  exit 1
fi

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; n=torch.cuda.device_count(); print(torch.cuda.get_device_name(0), 'ngpu', n); assert n==1, 'expected 1 GPU, got %s' % n"

# Mixed exact+fuzzy, one gold-centered crop, AraBERT-base, lr 1e-5, 20k claims.
# Full-page eval on 400 val (same protocol as the mBERT smoke). Does not overwrite v2.
python scripts/train_extractive_bert.py \
  --model aubmindlab/bert-base-arabertv2 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-5 \
  --max-train-claims 20000 \
  --eval-split val \
  --max-eval-claims 400 \
  --output-dir models/extractive_bert_v3
