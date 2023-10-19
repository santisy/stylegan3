#!/bin/bash
python measure_gen_metrics.py \
    --real_data datasets/lsunchurch_for_stylegan/ \
    --network_ae training_runs/en_0929_01/network-snapshot-002400.pkl \
    --network_diff training_runs/diff_1008_01/network-snapshot-7200.pkl \
    --exp_name diffv2_infer_try \
    --sample_total_img=128 \
    --generate_batch_size 64
