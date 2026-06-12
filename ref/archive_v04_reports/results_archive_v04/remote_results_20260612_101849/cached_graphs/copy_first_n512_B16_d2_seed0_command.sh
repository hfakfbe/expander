cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
for method in local random zigzag; do
  CUDA_VISIBLE_DEVICES="" /home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
  python cache_attention_graph.py \
    --method "$method" \
    --seq-len 512 \
    --block-size 16 \
    --degree 2 \
    --seed 0 \
    --output-dir ../../cached_graphs/copy_first_n512_B16_d2_seed0_${method}
done
