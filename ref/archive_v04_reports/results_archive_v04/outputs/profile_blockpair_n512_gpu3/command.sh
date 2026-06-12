cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES=3 /home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python profile_attention_backend.py \
  --task copy_first \
  --methods dense,local,random,zigzag \
  --attention-backend auto_blockpair \
  --seq-len 512 \
  --block-size 16 \
  --degree 2 \
  --batch-size 32 \
  --d-model 128 \
  --layers 8 \
  --heads 4 \
  --ffn-dim 256 \
  --num-values 4 \
  --learning-rate 0.001 \
  --warmup-steps 20 \
  --measure-steps 100 \
  --repeats 3 \
  --output-dir ../../outputs/profile_blockpair_n512_gpu3
