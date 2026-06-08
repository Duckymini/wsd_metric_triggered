#!/usr/bin/env bash
set -euo pipefail

cd /scratch/optimisation
export PYTHONPATH=/scratch/optimisation:${PYTHONPATH:-}
export HF_HOME=/scratch/hf-cache
export TRANSFORMERS_CACHE=/scratch/hf-cache/transformers
export HF_DATASETS_CACHE=/scratch/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 \
  --output notebooks/inference_out.ipynb \
  notebooks/inference.ipynb

echo "Done. Output saved to notebooks/inference_out.ipynb"
