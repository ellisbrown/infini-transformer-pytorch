#!/bin/sh

export CUDA_VISIBLE_DEVICES=1


python train.py \
    --batch_size 32