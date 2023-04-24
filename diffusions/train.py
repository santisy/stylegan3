"""
    Train the diffusion on extracted neural coordinates
"""

import click
import math
import os
import sys

import numpy as np
import torch
from torch.optim import Adam

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

sys.path.insert(0, '.')
import dnnlib
import legacy
from torch_utils import misc
from training.training_loop import save_image_grid
import torchvision.utils as tvu

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
              default=10, show_default=True)
@click.option('--sample_num', type=int,
              help='Sampling number',
              default=4, show_default=True)
@click.option('--record_k', type=int,
              help='How many k images to record loss',
              default=4, show_default=True)
@click.option('--snap_k', type=int,
              help='How many k images to save network snapshots',
              default=200, show_default=True)
def train_diffusion(
    exp_id: str,
    encoder_decoder_network: str,
    dataset: str,
    dim: int,
    feat_spatial_size: int,
    batch_size: int,
    train_lr: float,
    sample_k: int,
    sample_num: int,
    record_k: int,
    snap_k: int
):
    # The device
    device = torch.device('cuda')

    # Prepare folder and tensorboard
    run_dir = os.path.join('training_runs', exp_id)
    os.makedirs(run_dir, exist_ok=True)
    import torch.utils.tensorboard as tensorboard
    stats_tfevents = tensorboard.SummaryWriter(run_dir)
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'),
                       file_mode='a', should_flush=True)
    
    # The encoder decoder one
    with dnnlib.util.open_url(encoder_decoder_network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


    # Set randomness
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Diffusion Unet Module and optimizer
    d_unet = Unet(dim,
                  channels=G.feat_coord_dim).to(device)
    opt = Adam(d_unet.parameters(), lr=train_lr, betas = (0.9, 0.99))

    # Diffusion
    diffusion = GaussianDiffusion(
        d_unet,
        image_size = feat_spatial_size,
        timesteps = 1000,   # number of steps
        loss_type = 'l1',    # L1 or L2
        auto_normalize = False,
    ).to(device)

    # Dataset, sampler and iterator
    dataset = Dataset(dataset,
                      dim=G.feat_coord_dim,
                      size=feat_spatial_size)
    sampler = misc.InfiniteSampler(dataset)
    training_iter = iter(torch.utils.data.DataLoader(dataset=dataset,
                                                     sampler=sampler,
                                                     batch_size=batch_size))
    

    # Counting initials
    count = 0

    # Main Loop Starts Here --------------------
    while True:
        opt.zero_grad()
        real_nc = next(training_iter).to(device)
        loss = diffusion(real_nc)
        loss.backward()
        opt.step()

        # Increase couting
        count += batch_size
        global_step = count // 1000

        # Recording
        if count % (record_k * 1000) == 0:
            stats_tfevents.add_scalar('loss', loss.item(),
                                      global_step=global_step)

        # Sampling
        if count % (sample_k * 1000) == 0:
            sample_nc = diffusion.sample(sample_num)
            sample_imgs = decode_nc(G, sample_nc).cpu().numpy()
            # Save image to local target folder
            save_image_grid(sample_imgs,
                            os.path.join(run_dir,
                                         f'fakes{global_step:06d}.png'),
                            drange=[-1,1],
                            grid_size=(int(np.sqrt(sample_num)),
                                       int(np.sqrt(sample_num))))
            # Save image to tensorboard
            stats_tfevents.add_image(f'fake', tvu.make_grid(
                torch.tensor(sample_imgs[:16]),
                nrow=min(4, int(math.ceil(sample_imgs.shape[0]) ** 0.5)),
                normalize=True,
                value_range=(-1,1)
            ), global_step)

        # Save network snapshot
        if count % (snap_k * 1000) == 0:
            torch.save(d_unet.state_dict(),
                       os.path.join(run_dir,
                                    f'network-snapshot-{global_step}.pkl'))


if __name__ == '__main__':
    train_diffusion()
