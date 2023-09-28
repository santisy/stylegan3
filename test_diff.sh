#!/bin/bash
python infer_unconditional.py  \
    --encoder_decoder_network training_runs/en_0921_01/network-snapshot-002000.pkl \
    --exported_root exported \
    --output_dir training_runs/diff_0924_02 \
    --feat_spatial_size 64 \
    --batch_size 128 \
    --total_gen_nk 50 \
    --resume_from_checkpoint "latest" \
    --use_ema
