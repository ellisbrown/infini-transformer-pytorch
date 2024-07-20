
    # --num_processes 1 \
# accelerate launch \
#     train_accel_streaming.py \
#     --batch_size 8 \
#     --gradient_accumulate_every 16 \
#     --total_steps 1000 \
#     --depth 12 \
#     --dim 1024 \
#     --dim_head 128 \
#     --segment_len 1024 \
#     --seq_len 4096 \
#     --gen_seq_len 512 \
#     --val_steps 100 \
#     --dataset_name allenai/c4 \
#     --dataset_version en

accelerate launch \
    train_accel_streaming.py \
    --batch_size 4 \
    --gradient_accumulate_every 16 \
    --total_steps 1_000 \
    --depth 12 \
    --heads 8 \
    --dim 2048 \
    --dim_head 128 \
    --segment_len 2048 \
    --seq_len 32768 \
    --gen_seq_len 512 \
    --val_steps 100 \
    --dataset_name allenai/c4 \
    --dataset_version en



ok here's my current code. can you help me reorganize it around the number of steps instead of number of epochs?

with a massive dataset like this, don't want to wait until we've processed all of the data 