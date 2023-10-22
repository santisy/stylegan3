#!/bin/bash
python diffusions/train.py \
    --exp_id test_diff_cosine_variant \
    --batch_size 32 \
    --encoder_decoder_network training_runs/en_1011_01/network-snapshot-003607.pkl \
    --dataset datasets/lsunchurch_for_stylegan.zip \
    --dim 256 \
    --sample_num 16 \
    --record_k 1 \
    --train_lr 8e-5 \
    --feat_spatial_size 64 \
    --num_resnet_blocks '2,2,2,2' \
    --cosine_decay_max_steps 1000000 \
    --dim_mults '1,2,3,4' \
    --atten_layers '2,3,4' \
    --snap_k 1280 \
    --sample_k 1280 \
    --warmup_steps 10000 \
    --noise_scheduler cosine_variant \
    --no_noise_perturb false \
    --use_min_snr false \
    --copy_back true
    #--work_on_tmp_dir true