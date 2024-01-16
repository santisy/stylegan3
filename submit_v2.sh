#!/bin/bash
#SBATCH --time=160:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name="en_0105_01"
#SBATCH --output=./sbatch_logs/%j.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process (list hostnames of the nodes you've requested)
# NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l`
# echo NPROCS=$NPROCS

# Source the environment, load everything here
source ~/.bashrc

# Running training jobs
bash run.sh en_0105_01 2 24 \
    datasets/lsunchurch_total.zip \
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
    --en_lr_mult 0.1 \
    --pg_hash_res true \
    --pg_init_method "median" \
    --pg_init_iter_k 1000 \
    --pg_hr_iter_k 200 \
    --pg_detach true

#    --invert_coord true \
#    --glr 0.001 \
