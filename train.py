# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    #prev_run_dirs = []
    #if os.path.isdir(outdir):
    #    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    #prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    #prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    #cur_run_id = max(prev_run_ids, default=-1) + 1
    #c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    #assert not os.path.exists(c.run_dir)
    c.run_dir = outdir

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=True)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2', 'hashed']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str, required=True)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--img_snap',     help='How often to save image snapshots', metavar='TICKS',      type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# Hash setting related
@click.option('--res_min',      help='Minimum resolution level of hash maps.',                  type=int, default=16, show_default=True)
@click.option('--head_dim',     help='The dimension for attention head.',                       type=int, default=64, show_default=True)
@click.option('--table_num',    help='The number of hash tables.',                              type=int, default=16, show_default=True)
@click.option('--table_size_log2',  help='The log2 size of hash tables.',                       type=int, default=12, show_default=True)
@click.option('--z_dim',        help='The number of z dimension.',                              type=int, default=256, show_default=True)
@click.option('--mlp_hidden',   help='The mini mlp hidden dimension.',                          type=int, default=64, show_default=True)
@click.option('--init_dim',     help='Initial dimension/size of each hash table',               type=int, default=32, show_default=True)
@click.option('--style_dim',    help='Style dimension of the mapped latent code',               type=int, default=256, show_default=True)
@click.option('--use_layer_norm',   help='Use layer norm in transformer block.',                type=bool, default=False, show_default=True)
@click.option('--modulated_mini_linear', help='If the minilinear is modulated or not',          type=bool, default=True, show_default=True)
@click.option('--more_layer_norm',  help='Use layer norm in more linear space.',                type=bool, default=False, show_default=True)
@click.option('--fixed_random',     help='The upsample is fixed but randomized.',               type=bool, default=True,  show_default=True)
@click.option('--linear_up',    help='The upsample is linear or not.',                          type=bool, default=False, show_default=True)
@click.option('--output_skip',  help='Output skip true or not',                                 type=bool, default=False, show_default=True)
@click.option('--shuffle_input', help='Shuffle the input accoridng to random indices',          type=bool, default=False, show_default=True)
@click.option('--spatial_atten', help='Spatial attention or not.',                              type=bool, default=False, show_default=True)
@click.option('--two_style_code',   help='Using two stylecode or not.',                         type=bool, default=False, show_default=True)
@click.option('--tokenwise_linear', help='Use token wise linear or not.',                       type=bool, default=False, show_default=True)
@click.option('--shrink_down', help='Whether to shrink it down or not.',                        type=bool, default=False, show_default=True)
@click.option('--no_norm_layer', help='No normalization layer.',                                type=bool, default=False, show_default=True)
@click.option('--img_size',     help='resize_img to something else.',                           type=int, default=-1, show_default=True)
@click.option('--inter_filter', help='Among linear use inter conv/filter.',                     type=bool, default=False, show_default=True)
@click.option('--fixed_token_number', help='The number of token number is fixed.',              type=bool, default=False, show_default=True)
@click.option('--kernel_size', help='Kernel size of 1d convolution.',                           type=int, default=15, show_default=True)
@click.option('--init_res', help='Initial resolution retrieved from hash table.',               type=int, default=64, show_default=True)
@click.option('--level_dim', help='The dimension of each entry of hash table.',                 type=int, default=4, show_default=True)
@click.option('--feat_coord_dim', help='Style to hyperspace coordinate dimension.',             type=int, default=8, show_default=True)
@click.option('--dummy_hash_table', help='Dummy output of the hash table',                      type=bool, default=False, show_default=True)
@click.option('--tile_coord', help='If true all cordinates tiled once',                         type=bool, default=False, show_default=True)
@click.option('--discrete_all', help='Discretize all',                                          type=bool, default=False, show_default=True)
@click.option('--mini_linear_n_layers', help='Mini-linear n layers.',                           type=int, default=3, show_default=True)
@click.option('--eps_g', help='Epsilon of Adam optimizer of generator.',                        type=float, default=1e-15, show_default=True)
@click.option('--additional_first_shortcut', help='Additional first shortcut in block',         type=bool, default=False, show_default=True)


@click.option('--encoder_flag', help='Enable encoder training',                                 type=bool, default=False, show_default=True)
@click.option('--encoder_ch', help='Encoder base ch',                                           type=int, default=32, show_default=False)
@click.option('--l2loss_weight', help='MSE loss weight',                                        type=float, default=20.0, show_default=True)
@click.option('--one_hash_group', help='One hash group out.',                                   type=bool, default=False, show_default=True)
@click.option('--non_style_decoder', help='None style decoder.',                                type=bool, default=False, show_default=True)
@click.option('--context_coordinates', help='None average coordinates output.',                 type=bool, default=False, show_default=True)
@click.option('--concat_discriminator', help='Concatenate discriminator.',                      type=bool, default=False, show_default=True)
@click.option('--disable_patch_gan', help='Disable patch gan.',                                 type=bool, default=False, show_default=True)
@click.option('--feat_coord_dim_per_table', help='Feat coord per table.',                       type=int, default=2, show_default=True)
@click.option('--num_downsamples', help='Downsampling number of encoder.',                      type=int, default=5, show_default=True)
@click.option('--additional_decoder_conv', help='Additional decoder convolution.',              type=bool, default=False, show_default=True)
@click.option('--noise_perturb',    help='Noise perturbation of the neural coordinates.',       type=bool, default=False, show_default=True)
@click.option('--noise_perturb_sigma', help='Noise perturbation sigma value.',                  type=float, default=-1.0, show_default=True)
@click.option('--use_kl_reg',  help='Use KL regularization.',                                   type=bool, default=False, show_default=True)
@click.option('--kl_loss_weight', help='KL loss weight.',                                       type=float, default=1e-4, show_default=True)
@click.option('--split_val_n', help='The last n number images to be validation dataset.',       type=int, default=-1, show_default=True)
@click.option('--hash_res_ratio', help='The ratio of maximum hash resolution to init_res',      type=int, default=1, show_default=True)
@click.option('--expand_dim', help='Whether to expand the channel dimension',                   type=int, default=-1, show_default=True)
@click.option('--attn_resolutions', help='Where to do attention on encoder',                    
              type=lambda x: [int(y) for y in x.split(',')] if x is not None else None, default=None, show_default=True)
@click.option('--fused_spatial', help='Fused spatial localization y in hash and x learning.',   type=bool, default=False, show_default=True)
@click.option('--vq_decoder', help='Use vq gan decoder and patch gan discriminator',            type=bool, default=False, show_default=True)
@click.option('--circular_reuse', help='Circularly reuse the key codes',                        type=bool, default=False, show_default=True)
@click.option('--larger_decoder', help='Even larger decoder.',                                  type=bool, default=False, show_default=True)
@click.option('--encoder_ch', help='Encoder unit channel number.',                              type=int, default=32, show_default=True)
@click.option('--movq_decoder', help='Modulated VQ decoder',                                    type=bool, default=False, show_default=True)
@click.option('--encoder_resnet_num', help='The number of resnet layers.',                      type=int, default=4, show_default=False)
@click.option('--hash_resolution',    help='Force set hash table maximum resolution.',          type=int, default=-1, show_default=False)
@click.option('--no_concat_coord', help='Do not concate spatial coordinates.',                  type=bool, default=False, show_default=True)
@click.option('--local_coords',     help='Concat local coordinates.',                           type=bool, default=False, show_default=True)
@click.option('--combine_coords',    help='Combine spatial coordinates to one dimension.',      type=bool, default=False, show_default=True)
@click.option('--exhaustive_hash_sampling', help='Sampling all resolutions',                    type=bool, default=False, show_default=True)


def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name='training.hash_retrieve_generator.HashRetrieveGenerator',
                                 z_dim=opts.z_dim,
                                 res_min=opts.res_min,
                                 head_dim=opts.head_dim,
                                 table_num=opts.table_num,
                                 table_size_log2=opts.table_size_log2,
                                 mlp_hidden=opts.mlp_hidden,
                                 init_dim=opts.init_dim,
                                 style_dim=opts.style_dim,
                                 use_layer_norm=opts.use_layer_norm,
                                 modulated_mini_linear=opts.modulated_mini_linear,
                                 more_layer_norm=opts.more_layer_norm,
                                 fixed_random=opts.fixed_random,
                                 linear_up=opts.linear_up,
                                 output_skip=opts.output_skip,
                                 map_depth=opts.map_depth,
                                 shuffle_input=opts.shuffle_input,
                                 spatial_atten=opts.spatial_atten,
                                 two_style_code=opts.two_style_code,
                                 tokenwise_linear=opts.tokenwise_linear,
                                 shrink_down=opts.shrink_down,
                                 no_norm_layer=opts.no_norm_layer,
                                 inter_filter=opts.inter_filter,
                                 fixed_token_number=opts.fixed_token_number,
                                 kernel_size=opts.kernel_size,
                                 init_res=opts.init_res,
                                 level_dim=opts.level_dim,
                                 feat_coord_dim=opts.feat_coord_dim,
                                 dummy_hash_table=opts.dummy_hash_table,
                                 tile_coord=opts.tile_coord,
                                 discrete_all=opts.discrete_all,
                                 mini_linear_n_layers=opts.mini_linear_n_layers,
                                 one_hash_group=opts.one_hash_group,
                                 non_style_decoder=opts.non_style_decoder,
                                 context_coordinates=opts.context_coordinates,
                                 feat_coord_dim_per_table=opts.feat_coord_dim_per_table,
                                 num_downsamples=opts.num_downsamples,
                                 additional_decoder_conv=opts.additional_decoder_conv,
                                 noise_perturb=opts.noise_perturb,
                                 noise_perturb_sigma=opts.noise_perturb_sigma,
                                 use_kl_reg=opts.use_kl_reg,
                                 hash_res_ratio=opts.hash_res_ratio,
                                 expand_dim=opts.expand_dim,
                                 attn_resolutions=opts.attn_resolutions,
                                 fused_spatial=opts.fused_spatial,
                                 vq_decoder=opts.vq_decoder,
                                 circular_reuse=opts.circular_reuse,
                                 larger_decoder=opts.larger_decoder,
                                 encoder_ch=opts.encoder_ch,
                                 movq_decoder=opts.movq_decoder,
                                 encoder_resnet_num=opts.encoder_resnet_num,
                                 hash_resolution=opts.hash_resolution,
                                 no_concat_coord=opts.no_concat_coord,
                                 local_coords=opts.local_coords,
                                 combine_coords=opts.combine_coords,
                                 exhaustive_hash_sampling=opts.exhaustive_hash_sampling,
                                 )
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=opts.eps_g)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.img_size = opts.img_size
    c.training_set_kwargs.split_val_n = opts.split_val_n

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.D_kwargs.channel_base = opts.cbase
    c.D_kwargs.channel_max = opts.cmax
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = 0.002 #(0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.network_snapshot_ticks = opts.snap
    c.image_snapshot_ticks = opts.img_snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Encoder args
    c.encoder_flag = opts.encoder_flag
    c.loss_kwargs.encoder_flag = opts.encoder_flag
    c.loss_kwargs.l2loss_weight = opts.l2loss_weight
    c.loss_kwargs.use_kl_reg = opts.use_kl_reg
    c.loss_kwargs.kl_loss_weight = opts.kl_loss_weight
    c.loss_kwargs.vq_decoder = opts.vq_decoder
    if opts.encoder_flag:
        c.G_kwargs.class_name = 'training.hash_autoencoder_generator.HashAutoGenerator'
    c.D_kwargs.encoder_flag = opts.encoder_flag
    c.D_kwargs.disable_patch_gan = opts.disable_patch_gan

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    if opts.vq_decoder:
        # Now use the default settings
        c.D_kwargs = dnnlib.EasyDict(class_name='training.discriminator.NLayerDiscriminator')
        c.D_reg_interval = None

    # Description string, fully use the given one. Other information will 
    #   be recorded in a sheet.
    desc = f'{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
