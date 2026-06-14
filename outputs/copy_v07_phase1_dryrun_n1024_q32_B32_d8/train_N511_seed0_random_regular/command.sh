#!/usr/bin/env bash
set -euo pipefail
cd /Users/sxye/Documents/expander
conda activate ysx_base 2>/dev/null || true
COPY_V06_GIT_COMMIT='' CUDA_VISIBLE_DEVICES='' /Users/sxye/miniconda3/bin/python scripts/run_experiment.py --config configs/copy_v07_smoke_n1024_q32_B32_d8.json --output-dir outputs/copy_v07_phase1_dryrun_n1024_q32_B32_d8 --methods local,zigzag_certified,random_regular --steps 1 --batch-size 1 --eval-batches 1 --d-model 16 --layers 1 --heads 1 --ffn-dim 32 --dropout 0 --device cpu --skip-tests
