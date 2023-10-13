#!/bin/bash
# I suspect the batch size is not enough for the original diffusion training
accelerate launch --main_process_port 29502 \
    diffusions/train.py \
    --exp_id test_diffv2_chen_linear_without_minsnr_1012_01 \
    --batch_size 12 \
    --encoder_decoder_network training_runs/en_0908_01/network-snapshot-002600.pkl \
    --dataset datasets/lsunchurch_total.zip \
    --dim 256 \
    --sample_num 16 \
    --record_k 1 \
    --train_lr 6e-5 \
    --feat_spatial_size 64 \
    --num_resnet_blocks '2,2,2,2' \
    --no_noise_perturb true \
    --cosine_decay_max_steps 1000000 \
    --dim_mults '1,2,3,4' \
    --atten_layers '2,3,4' \
    --snap_k 160 \
    --sample_k 32 \
    --warmup_steps 10000 \
    --noise_scheduler chen_linear \
    --use_min_snr false
