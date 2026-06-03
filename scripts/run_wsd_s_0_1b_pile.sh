#!/usr/bin/env bash
set -euo pipefail

cd /scratch/optimisation
export PYTHONPATH=/scratch/optimisation:${PYTHONPATH:-}

python3 -m src.train --config configs/wsd_s_0_1b_pile.yaml
