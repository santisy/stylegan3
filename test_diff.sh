#!/bin/bash
# I suspect the batch size is not enough for the original diffusion training
bash run.sh test_en 1 16 ./datasets/lsunchurch_for_stylegan.zip \
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
    --grid_type="tiled" \
    --noise_perturb=true \
    --attn_resolutions 64
