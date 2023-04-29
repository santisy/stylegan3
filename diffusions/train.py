"""
    Train the diffusion on extracted neural coordinates
"""

import click
import json
import math
import os
import sys
import time

import numpy as np
import torch
sys.path.insert(0, '.')

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from ldm.models.diffusion.ddpm import DDPM

import dnnlib
import legacy
from torch_utils import misc
from training.training_loop import save_image_grid
import torchvision.utils as tvu

from utils.utils import delete_file
from diffusions.dataset import Dataset
from diffusions.decode import decode_nc




@click.command()
@click.option('--exp_id', type=str, help='The experiment ID', required=True)
@click.option('--encoder_decoder_network', type=str,
              help='The encoder decoder network pkl path string.', required=True)
@click.option('--dataset', type=str, required=True,
              help='Dataset folder path that contains the npy files.')
@click.option('--batch_size', type=int, default=8, show_default=True)
@click.option('--dim', type=int, help='Unit dimension of the UNet.',
              default=128, show_default=True)
@click.option('--feat_spatial_size', type=int, default=16, show_default=True)
@click.option('--train_lr', type=float, help='The learning rate',
              default=1e-4)
@click.option('--sample_k', type=int,
              help='How many k images to sample a result and decode',
              default=50, show_default=True)
@click.option('--sample_num', type=int,
              help='Sampling number',
              default=5, show_default=True)
@click.option('--record_k', type=int,
              help='How many k images to record loss',
              default=5, show_default=True)
@click.option('--snap_k', type=int,
              help='How many k images to save network snapshots',
              default=2000, show_default=True)
def train_diffusion(**kwargs):

    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    # The device
    device = torch.device('cuda')

    # Prepare folder and tensorboard
    run_dir = os.path.join('training_runs', opts.exp_id)
    os.makedirs(run_dir, exist_ok=True)
    import torch.utils.tensorboard as tensorboard
    stats_tfevents = tensorboard.SummaryWriter(run_dir)
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'),
                       file_mode='a', should_flush=True)
    
    # Dump config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(opts, f, indent=4)
    
    # The encoder decoder one
    with dnnlib.util.open_url(opts.encoder_decoder_network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = G.eval()


    # Set randomness
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Diffusion Unet Module and optimizer --------------------
    unet = Unet(dim=opts.dim,
                channels=G.feat_coord_dim,
                dim_mults=(1, 2, 4),
                num_resnet_blocks=(2, 4, 8),
                layer_attns=(False, False, True),
                use_linear_attn=False,
                cond_on_text=False
                )
    imagen = Imagen(
            condition_on_text = False,
            unets = (unet, ),
            image_sizes = (opts.feat_spatial_size, ),
            timesteps = 1000,
            channels=G.feat_coord_dim,
            auto_normalize_img=True,
            min_snr_gamma=5,
            min_snr_loss_weight=True,
            dynamic_thresholding=False,
            pred_objectives='noise', # noise or x_start
            )
    trainer = ImagenTrainer(imagen=imagen,
                            imagen_checkpoint_path=None, # TODO: continue training
                            lr=opts.train_lr,
                            cosine_decay_max_steps=1e12,  # Note I manually change the eta_min to 1e-5
                            ).to(device)

    # ------------------------------

    # Dataset, sampler and iterator
    dataset = Dataset(opts.dataset,
                      dim=G.feat_coord_dim,
                      size=opts.feat_spatial_size)
    sampler = misc.InfiniteSampler(dataset)
    training_iter = iter(torch.utils.data.DataLoader(dataset=dataset,
                                                     sampler=sampler,
                                                     batch_size=opts.batch_size,
                                                     prefetch_factor=2))
    

    # Counting initials
    count = 0
    save_snapshot_list = []
    start_time = time.time()
    tick_end_time = None
    scale_std = None

    # Main Loop Starts Here --------------------
    while True:
        # Get data and forward
        real_ni = next(training_iter)
        loss = trainer(real_ni, unet_number=1)
        trainer.update(unet_number = 1)

        # Increase couting
        count += opts.batch_size
        global_step = count // 1000

        # Recording
        if count % (opts.record_k * 1000) == 0:
            stats_tfevents.add_scalar('loss', loss,
                                      global_step=global_step)
            tick_end_time = time.time()
            print(f'step {global_step}\t'
                  f'loss {loss:.2f}\t'
                  f'time {dnnlib.util.format_time(tick_end_time - start_time)}\t')

        # Sampling
        if count % (opts.sample_k * 1000) == 0:
            print('Save image ...')
            sample_ni = trainer.sample(batch_size=opts.sample_num, use_tqdm=False)
            print('Value range of sampled nc: ', sample_ni.min(), sample_ni.max())
            print('Stats of sampled nc:', sample_ni.mean(), sample_ni.var())
            sample_ni = torch.clip(sample_ni, 0, 1)
            with torch.no_grad():
                sample_imgs = decode_nc(G, sample_ni).cpu().numpy()
            # Save image to local target folder
            save_image_grid(sample_imgs,
                            os.path.join(run_dir,
                                         f'fakes{global_step:06d}.png'),
                            drange=[-1,1],
                            grid_size=(int(np.sqrt(opts.sample_num)),
                                       int(np.sqrt(opts.sample_num))))
            # Save image to tensorboard
            stats_tfevents.add_image(f'fake', tvu.make_grid(
                torch.tensor(sample_imgs[:16]),
                nrow=min(4, int(math.ceil(sample_imgs.shape[0]) ** 0.5)),
                normalize=True,
                value_range=(-1,1)
            ), global_step)

        # Save network snapshot
        if count % (opts.snap_k * 1000) == 0:
            print(f'Save network-snapshot-{global_step}.pkl ...')
            save_file = os.path.join(run_dir, f'network-snapshot-{global_step}.pkl')
            save_snapshot_list.append(save_file)
            trainer.save(save_file)
            if len(save_snapshot_list) > 5:
                delete_file(save_snapshot_list.pop(0))


if __name__ == '__main__':
    train_diffusion()
