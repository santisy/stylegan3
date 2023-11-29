#!/bin/bash
# Source the environment, load everything here
# list out some useful information (optional)

echo "Node $SLURM_NODEID says : SLURMTMPDIR="$SLURM_TMPDIR
echo "Node $SLURM_NODEID says: main node at $HEAD_NODE"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."
# sample process (list hostnames of the nodes you've requested)
source ~/.bashrc

# Running training jobs
torchrun \
    --master-port $HEAD_NODE_PORT \
    --master-addr $HEAD_NODE  \
    --nnodes $SLURM_NNODES \
    --nproc-per-node 1 \
    --node-rank $SLURM_NODEID \
    --no-python \
    run.sh en_1125_01 2 24 \
    datasets/lsunchurch_for_stylegan.zip \
    --gamma=4 \
    --table_size_log2=22 \
    --level_dim=4 \
    --feat_coord_dim=4 \
    --img_snap=2 \
    --init_res=64 \
    --style_dim=512 \
    --img_size=256 \
    --table_num=16 \
    --res_min=16 \
    --init_dim=512 \
    --tile_coord=true \
    --encoder_flag=true \
    --mini_linear_n_layers=3 \
    --disable_patch_gan=true \
    --feat_coord_dim_per_table=1 \
    --num_downsamples=3 \
    --additional_decoder_conv=true \
    --use_kl_reg=false \
    --noise_perturb=false \
    --hash_res_ratio 16 \
    --encoder_ch 64 \
    --align_corners true \
    --grid_type "tiled" \
    --pg_hash_res true \
    --pg_init_method 'none' \
    --pg_hr_iter_k 100 \
    --pg_alter_opti true