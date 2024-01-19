bash run.sh test_3d 1 1 \
    datasets/chair_sdf.zip \
    --gamma=4 \
    --table_size_log2=22 \
    --level_dim=2 \
    --feat_coord_dim=4 \
    --img_snap=20 \
    --init_res=32 \
    --style_dim=512 \
    --img_size=256 \
    --table_num=16 \
    --res_min=8 \
    --init_dim=128 \
    --tile_coord=true \
    --encoder_flag=true \
    --mini_linear_n_layers=3 \
    --disable_patch_gan=true \
    --feat_coord_dim_per_table=1 \
    --num_downsamples=4 \
    --additional_decoder_conv=true \
    --use_kl_reg=false \
    --noise_perturb=false \
    --encoder_ch 16 \
    --encoder_resnet_num 1 \
    --decoder_resnet_num 1 \
    --hash_res_ratio 4 \
    --align_corners true \
    --grid_type "tiled" \
    --pg_hash_res true \
    --pg_init_method "median" \
    --pg_init_iter_k 100 \
    --pg_hr_iter_k 50 \
    --en_lr_mult 0.1 \
    --pg_detach true \
    --flag_3d true \
    --resout_flag true \
    --no_pre_pixelreshape true