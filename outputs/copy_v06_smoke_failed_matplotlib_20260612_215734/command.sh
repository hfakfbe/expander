#!/usr/bin/env bash
set -euo pipefail
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base 2>/dev/null || true
COPY_V06_GIT_COMMIT='' CUDA_VISIBLE_DEVICES=3 /home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/synthetic_mvp.py --config configs/copy_v06_smoke.json --output-dir outputs/copy_v06_smoke
