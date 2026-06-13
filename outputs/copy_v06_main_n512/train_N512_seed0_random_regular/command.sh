#!/usr/bin/env bash
set -euo pipefail
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base 2>/dev/null || true
COPY_V06_GIT_COMMIT=525791fb5f020146bbd8aee3400c3ac7fb998521 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/huiwei/ysx/zigzag_attention/.deps /home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/synthetic_mvp.py --config configs/copy_v06_main_n512.json --output-dir outputs/copy_v06_main_n512
