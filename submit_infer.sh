#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="infer2_diff_1001_01"
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

# !/bin/bash
python infer_unconditional.py  \
    --encoder_decoder_network training_runs/en_0925_01/network-snapshot-002400.pkl \
    --exported_root exported \
    --output_dir training_runs/diff_1001_01 \
    --feat_spatial_size 64 \
    --batch_size 128 \
    --total_gen_nk 50 \
    --resume_from_checkpoint "latest" \
    --seed_start 500 \
    --use_ema
