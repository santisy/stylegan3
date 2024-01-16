#!/bin/bash
#SBATCH --time=23:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="infer11_diff_1114_01"
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
python measure_gen_metrics.py \
    --real_data datasets/lsunchurch_for_stylegan/ \
    --network_ae training_runs/en_1011_02/network-snapshot-000800.pkl \
    --network_diff training_runs/diff_1114_01/network-snapshot-5760.pkl \
    --diff_config training_runs/diff_1114_01/config.json \
    --exp_name diff_1114_01_5760 \
    --only_gen true \
    --seed 11 \
    --sample_total_img 13000 \
    --generate_batch_size 128
