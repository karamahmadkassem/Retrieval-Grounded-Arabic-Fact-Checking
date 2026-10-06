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
mkdir -p logs models/extractive_bert

module purge
module load cuda
module load python/ai-4
export PYTHONUNBUFFERED=1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# Full train, 400-claim val (full val is ~8h of eval and will TIMEOUT).
python scripts/train_extractive_bert.py \
  --model bert-base-multilingual-cased \
  --epochs 1 \
  --batch-size 8 \
  --eval-split val \
  --max-eval-claims 400
