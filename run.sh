#!/bin/bash
python \
    train.py \
    --desc=${1} \
    --gpus=${2} \
    --batch=${3} \
    --outdir=training_runs/${1} \
    --cfg='hashed' \
    --data=${4} \
    ${@:5}
