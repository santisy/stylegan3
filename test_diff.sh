#!/bin/bash
python \
    diffusions/train.py \
    --exp_id test_diff_3d \
    --batch_size 16 \
    --encoder_decoder_network training_runs/en3d_1210_01/network-snapshot-000400.pkl \
    --dataset datasets/chair_sdf.zip \
    --dim 128 \
    --sample_num 4 \
    --record_k 1 \
    --train_lr 8e-5 \
    --feat_spatial_size 16 \
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
    --flag_3d true \
    --work_on_tmp_dir true
