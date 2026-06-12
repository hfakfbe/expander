cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES="" \
/home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python profile_attention_backend.py \
  --methods local,zigzag \
  --seq-len 64 \
  --block-size 8 \
  --degree 2 \
  --warmup-steps 1 \
  --measure-steps 2 \
  --repeats 1 \
  --batch-size 2 \
  --d-model 32 \
  --layers 1 \
  --heads 2 \
  --ffn-dim 64 \
  --attention-backend auto_split \
  --output-dir ../../outputs/profile_cpu_smoke
