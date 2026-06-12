#!/usr/bin/env bash
set -euo pipefail
cd /Users/sxye/Documents/expander
conda activate ysx_base 2>/dev/null || true
COPY_V05_GIT_COMMIT='' CUDA_VISIBLE_DEVICES='' /Users/sxye/miniconda3/bin/python scripts/synthetic_mvp.py --config configs/copy_v05_smoke.json --output-dir outputs/copy_v05_smoke_local
