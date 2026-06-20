#!/usr/bin/env bash
set -euo pipefail

cd /home/huiwei/ysx/zigzag_attention
PYTHON=/home/huiwei/miniconda3/envs/ysx_base/bin/python
LOG_DIR=logs/copy_corrected_q32_B64_d32_l8_log5_random_density_sweep
mkdir -p "$LOG_DIR"

configs=(
  configs/copy_corrected_q32_B64_d32_l8_log5_random_density50.json
  configs/copy_corrected_q32_B64_d32_l8_log5_random_density80.json
  configs/copy_corrected_q32_B64_d32_l8_log5_random_density90.json
)

for cfg in "${configs[@]}"; do
  trial="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_id"])' "$cfg")"
  run_dir="outputs/copy_corrected_q32_B64_d32_l8_log5/runs/$trial/random_regular/seed0"
  train_log="$LOG_DIR/${trial}_train.log"
  eval_log="$LOG_DIR/${trial}_final_eval.log"
  echo "$(date -Is) start train cfg=$cfg trial=$trial"
  if [[ -s "$run_dir/summary.json" ]]; then
    echo "$(date -Is) existing summary found; runner will validate identity or refuse stale skip"
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_copy_corrected.py \
    --config "$cfg" --mode train --method random_regular --seed 0 --device cuda \
    2>&1 | tee "$train_log"
  echo "$(date -Is) start final-eval cfg=$cfg trial=$trial"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_copy_corrected.py \
    --config "$cfg" --mode final-eval --method random_regular --seed 0 --device cuda \
    2>&1 | tee "$eval_log"
  echo "$(date -Is) done trial=$trial"
done

"$PYTHON" scripts/summarize_copy_random_density_sweep.py
