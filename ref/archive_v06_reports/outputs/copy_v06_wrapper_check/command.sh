#!/usr/bin/env bash
set -euo pipefail
cd /Users/sxye/Documents/expander
conda activate ysx_base 2>/dev/null || true
COPY_V06_GIT_COMMIT='' CUDA_VISIBLE_DEVICES='' /Users/sxye/miniconda3/bin/python scripts/synthetic_mvp.py --config configs/copy_v06_smoke.json --output-dir outputs/copy_v06_wrapper_check --methods local --steps 1 --eval-batches 1 --batch-size 1 --device cpu --skip-tests
