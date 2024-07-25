
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

# export CUDA_VISIBLE_DEVICES=0 &&
# python \
accelerate launch \
    train_accel_streaming.py \
    --batch_size 2 \
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

    # --tokenizer_name tiktoken-cl100k_base \
    # --dataset_name pg19 \
    # --dataset_version None

