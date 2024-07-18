#!/bin/sh

export CUDA_VISIBLE_DEVICES=1


python train.py \
    --batch_size 32 \
    --depth 12 \
    --dim 1024 \
    --dim_head 128 \


