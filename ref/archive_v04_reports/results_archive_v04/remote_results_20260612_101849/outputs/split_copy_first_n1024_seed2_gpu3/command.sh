cd /home/huiwei/ysx/zigzag_attention/code/project_scripts
CUDA_VISIBLE_DEVICES=3 /home/huiwei/miniconda3/bin/conda run --no-capture-output -n ysx_base \
python synthetic_mvp.py \
  --task copy_first \
  --methods dense,local,random,zigzag \
  --seq-len 1024 \
  --block-size 16 \
  --degree 2 \
  --steps 1000 \
  --eval-batches 20 \
  --batch-size 16 \
  --d-model 128 \
  --layers 8 \
  --heads 4 \
  --ffn-dim 256 \
  --num-values 4 \
  --learning-rate 0.001 \
  --log-every 250 \
  --attention-backend auto_split \
  --seed 2 \
  --output-dir ../../outputs/split_copy_first_n1024_seed2_gpu3
