#!/bin/bash
#SBATCH --time=160:00:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=64G
#SBATCH --job-name="real_1019_03"
#SBATCH --output=./sbatch_logs/%j.log

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory = "$SLURM_SUBMIT_DIR
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l`
echo NPROCS=$NPROCS

export HEAD_NODE=$(hostname)
export HEAD_NODE_PORT=29513

chmod +x srun_diffv2.sh && srun srun_diffv2.sh