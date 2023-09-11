#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="test_buffterfly"
#SBATCH --output=./sbatch_logs/%j.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l`
echo NPROCS=$NPROCS

# Source the environment, load everything here
source ~/.bashrc

# Running training jobs
bash run.sh en_0908_01 1 16 ./datasets/lsunchurch_for_stylegan.zip \
    --gamma=4 \
    --table_size_log2=18 \
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
    --num_downsamples=2 \
    --additional_decoder_conv=true \
    --use_kl_reg=false \
    --noise_perturb=true \
    --attn_resolutions 64 \
    --grid_type='tiled'
