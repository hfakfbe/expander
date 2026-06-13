#!/usr/bin/env bash
set -euo pipefail
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base 2>/dev/null || true
COPY_V05_GIT_COMMIT=75330e074f67e3f181dc3e4cab2d941eb54dbf2d CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/huiwei/ysx/zigzag_attention/.deps /home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/synthetic_mvp.py --config configs/copy_v05_smoke.json --methods dense,local,random,zigzag --steps 2 --batch-size 2 --eval-batches 1 --output-dir outputs/copy_v05_phase1_smoke
