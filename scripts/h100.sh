


# accelerate launch \
/nfs/ellisb/infini-transformer-pytorch/env/bin/accelerate launch \
    --mixed_precision bf16 \
    train_accel.py \
    --batch_size 16 \
    --gradient_accumulate_every 16 \
    --num_epochs 100 \
    --depth 12 \
    --dim 1024 \
    --dim_head 128 \
    --segment_len 1024 \
    --seq_len 4096 \
    --gen_seq_len 512 \
    --wandb_run_name "h100_pg19" \
    --dataset_name pg19 \
    --dataset_version None \

    # --dataset_version wikitext-103-raw-v1