#!/bin/bash
#python \
#accelerate launch --main_process_port 29502 \
#    train_unconditional.py  \
#    --encoder_decoder_network training_runs/en_0929_05/network-snapshot-002400.pkl \
#    --train_data datasets/imagenet_train.zip \
#    --output_dir training_runs/test_diff_cond \
#    --feat_spatial_size 64 \
#    --train_batch_size 48 \
#    --num_epochs 400 \
#    --save_model_epochs 20 \
#    --save_images_steps_k 1 \
#    --learning_rate 1e-4 \
#    --use_ema \
#    --checkpoints_total_limit 5 \
#    --dataloader_num_workers 2 \
#    --checkpointing_steps 2000 \
#    --work_on_tmp_dir \
#    --class_condition


#accelerate launch --main_process_port 29502 \
#    diffusions/train.py \
#    --exp_id diff_1001_02 \
#    --batch_size 32 \
#    --encoder_decoder_network training_runs/en_0929_01/network-snapshot-002400.pkl \
#    --dataset datasets/lsunchurch_total.zip \
#    --dim 256 \
#    --sample_num 16 \
#    --record_k 1 \
#    --train_lr 8e-5 \
#    --feat_spatial_size 64 \
#    --num_resnet_blocks '2,2,2,2' \
#    --no_noise_perturb true \
#    --cosine_decay_max_steps 1000000 \
#    --dim_mults '1,2,3,4' \
#    --atten_layers '2,3,4' \
#    --snap_k 320 \
#    --sample_k 1280 \
#    --work_on_tmp_dir true \
#    --resume training_runs/diff_1001_02/network-snapshot-2560.pkl
#

accelerate launch --main_process_port 29502 \
    train_unconditional.py  \
    --encoder_decoder_network training_runs/en_0929_01/network-snapshot-002400.pkl \
    --train_data datasets/lsunchurch_total.zip \
    --output_dir training_runs/diff_1001_01 \
    --feat_spatial_size 64 \
    --train_batch_size 48 \
    --num_epochs 400 \
    --save_model_epochs 20 \
    --save_images_steps_k 4 \
    --learning_rate 1e-4 \
    --use_ema \
    --checkpoints_total_limit 5 \
    --dataloader_num_workers 2 \
    --checkpointing_steps 2000 \
    --work_on_tmp_dir \
    --resume_from_checkpoint "latest"
