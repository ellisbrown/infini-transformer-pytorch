

# accelerate launch train_accel.py \
#     --batch_size 8 \
#     --gradient_accumulate_every 16 \
#     --num_epochs 100 \
#     --depth 12 \
#     --dim 1024 \
#     --dim_head 128 \
#     --segment_len 1024 \
#     --seq_len 4096 \
#     --gen_seq_len 512 \
#     --dataset_version wikitext-103-raw-v1

accelerate launch train_accel.py \
    --batch_size 8 \
    --gradient_accumulate_every 16 \
    --num_epochs 100 \
    --depth 12 \
    --dim 1024 \
    --dim_head 128 \
    --segment_len 1024 \
    --seq_len 4096 \
    --gen_seq_len 512 \
    --dataset_name deepmind/pg19 \
    --dataset_version None