# python train.py \
accelerate launch train_accel.py \
    --batch_size 4 \
    --depth 12 \
    --dim 1024 \
    --dim_head 128 \
    --segment_len 1024 \
    --seq_len 4096 \
    --gen_seq_len 2048 \
