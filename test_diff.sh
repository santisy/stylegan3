#!/bin/bash
accelerate launch --main_process_port 29502 train_unconditional.py \
    --encoder_decoder_network training_runs/en_0908_01/network-snapshot-002600.pkl \
    --train_data_dir exported/en_0908_01_extracted \
    --output_dir training_runs/test_diff \
    --feat_spatial_size 64 \
    --train_batch_size 32 \
    --num_epochs 1000 \
    --save_model_epochs 20 \
    --save_images_steps_k 4 \
    --learning_rate 1e-4 \
    --use_ema \
    --checkpoints_total_limit 5 \
    --checkpointing_steps 2000