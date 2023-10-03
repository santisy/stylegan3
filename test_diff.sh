#!/bin/bash
python \
    train_unconditional.py  \
    --encoder_decoder_network training_runs/en_0929_05/network-snapshot-002400.pkl \
    --train_data datasets/imagenet_train.zip \
    --output_dir training_runs/test_diff_cond \
    --feat_spatial_size 64 \
    --train_batch_size 48 \
    --num_epochs 400 \
    --save_model_epochs 20 \
    --save_images_steps_k 1 \
    --learning_rate 1e-4 \
    --use_ema \
    --checkpoints_total_limit 5 \
    --dataloader_num_workers 2 \
    --checkpointing_steps 2000 \
    --work_on_tmp_dir \
    --class_condition
