#!/bin/bash
#SBATCH --time=160:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name="diff_1027_01"
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
    --encoder_decoder_network training_runs/en_0929_05_re/network-snapshot-002605.pkl \
    --train_data datasets/imagenet_train.zip \
    --output_dir training_runs/diff_1027_01 \
    --feat_spatial_size 64 \
    --train_batch_size 48 \
    --num_epochs 400 \
    --save_model_epochs 20 \
    --save_images_steps_k 20 \
    --learning_rate 9.5e-5 \
    --use_ema \
    --checkpoints_total_limit 5 \
    --dataloader_num_workers 2 \
    --checkpointing_steps 10000 \
    --work_on_tmp_dir \
    --ddpm_beta_schedule cosine_variant_v2 \
    --copy_back true \
    --no_noise_perturb false
