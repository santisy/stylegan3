#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name="diff_1003_01"
#SBATCH --output=./sbatch_logs/%j.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l`
echo NPROCS=$NPROCS

# Source the environment, load everything here
source ~/.bashrc

# Running training jobs
accelerate launch --main_process_port 29502 \
    train_unconditional.py  \
    --encoder_decoder_network training_runs/en_0929_05/network-snapshot-002400.pkl \
    --train_data datasets/imagenet_train.zip \
    --output_dir training_runs/diff_1003_01 \
    --feat_spatial_size 64 \
    --train_batch_size 48 \
    --num_epochs 200 \
    --save_model_epochs 1 \
    --save_images_steps_k 8 \
    --learning_rate 1e-4 \
    --use_ema \
    --checkpoints_total_limit 5 \
    --dataloader_num_workers 2 \
    --checkpointing_steps 8000 \
    --work_on_tmp_dir \
    --class_condition