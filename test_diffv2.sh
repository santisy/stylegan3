#!/bin/bash
python \
    diffusions/train.py \
    --exp_id test_diff_2d \
    --batch_size 48 \
    --encoder_decoder_network training_runs/en_1204_01_c/network-snapshot-009619.pkl \
    --dataset datasets/lsunchurch_total.zip \
    --dim 448 \
    --sample_num 16 \
    --record_k 1 \
    --train_lr 8e-5 \
    --feat_spatial_size 32 \
    --num_resnet_blocks '2,2,2,2' \
    --cosine_decay_max_steps 1000000 \
    --dim_mults '1,2,3,4' \
    --atten_layers '2,3,4' \
    --snap_k 960 \
    --sample_k 960 \
    --warmup_steps 10000 \
    --noise_scheduler cosine_variant_v2 \
    --no_noise_perturb true \
    --use_min_snr false \
    --copy_back true \
    --work_on_tmp_dir true
