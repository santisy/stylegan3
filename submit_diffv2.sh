#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name="diff_1001_02"
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
accelerate launch --main_process_port 29502 \
    diffusions/train.py \
    --exp_id diff_1001_02 \
    --batch_size 32 \
    --encoder_decoder_network training_runs/en_0929_01/network-snapshot-002400.pkl \
    --dataset datasets/lsunchurch_total \
    --dim 256 \
    --sample_num 16 \
    --record_k 1 \
    --train_lr 8e-5 \
    --feat_spatial_size 64 \
    --num_resnet_blocks '2,2,2,2' \
    --no_noise_perturb true \
    --cosine_decay_max_steps 1000000 \
    --dim_mults '1,2,3,4' \
    --atten_layers '2,3,4' \
    --snap_k 640 \
    --sample_k 1280