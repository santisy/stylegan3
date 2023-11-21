#!/bin/bash
# Source the environment, load everything here
# list out some useful information (optional)

echo "Node $SLURM_NODEID says : SLURMTMPDIR="$SLURM_TMPDIR
echo "Node $SLURM_NODEID says: main node at $HEAD_NODE"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."
# sample process (list hostnames of the nodes you've requested)
source ~/.bashrc

# Running training jobs
accelerate launch \
    --multi_gpu \
    --gpu_ids="all" \
    --main_process_port $HEAD_NODE_PORT \
    --main_process_ip $HEAD_NODE  \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --num_processes $(($SLURM_GPUS_PER_NODE*$SLURM_NNODES)) \
    diffusions/train.py \
    --exp_id diff_1019_03 \
    --batch_size 32 \
    --encoder_decoder_network training_runs/en_1011_01/network-snapshot-003607.pkl \
    --dataset datasets/lsunchurch_total.zip \
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
    --no_noise_perturb true \
    --use_min_snr false \
    --copy_back true \
    --work_on_tmp_dir true