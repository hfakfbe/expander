#!/usr/bin/env bash
set -euo pipefail
cd /home/huiwei/ysx/zigzag_attention
conda activate ysx_base 2>/dev/null || true
COPY_V06_GIT_COMMIT=a143e01 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages /home/huiwei/miniconda3/envs/ysx_base/bin/python scripts/run_experiment.py --config configs/copy_v07_main_n1024_q32_B32_d8.json --methods zigzag_certified_cosine --output-dir outputs/copy_v07_main_n1024_q32_B32_d8_parallel_zigzag_certified_cosine --device cuda --local-or-remote remote --log-path logs/copy_v07_main_n1024_q32_B32_d8_parallel_zigzag_certified_cosine_20260614T125736Z.log --skip-tests
