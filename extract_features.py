# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os

import click
import dnnlib
import numpy as np
import torch
import tqdm

import legacy
from utils.simple_dataset import SimpleDataset


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the neural coordinates', type=str, required=True, metavar='DIR')
@click.option('--dataset', help='Dataset zip file.', type=str, required=True)
@click.option('--dataset2', help='Another dataset (Usually the validation dataset.).', type=str, default='', show_default=True)
def generate_images(
    network_pkl: str,
    outdir: str,
    dataset: str,
    dataset2: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --truni=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uniurated images with truniation using the MetFaces-U dataset
    python gen_images.py --outdir=out --truni=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = G.eval()

    os.makedirs(outdir, exist_ok=True)
    print(f'\033[92mExtract to folder {outdir}\033[00m')

    # Construct the dataset
    simple_data_iter = iter(SimpleDataset(dataset, device))
    simple_data_iter2 = iter(SimpleDataset(dataset2, device)) if dataset2 != '' else None

    # Pbar and count
    count = 0
    pbar = tqdm.tqdm(total=len(simple_data_iter) if
                     simple_data_iter2 is None else
                     len(simple_data_iter2) + len(simple_data_iter))

    def iter_through(data_iter, count=0):
        # Iter through:
        for i, img in enumerate(data_iter):
            # Neural Coordinates Output
            with torch.no_grad():
                ni = G.img_encoder(img)
            # Check the shape
            if i == 0:
                print('ni shape in this run is', ni.shape)
            # Dump the data to folder
            ni = ni.cpu().numpy()
            out_file = os.path.join(outdir, f'{count:07d}.npz')
            with open(out_file, 'wb') as f:
                np.savez(f, ni=ni)
            pbar.update(1)
            count += 1
        return count

    # Iter through first dataset
    count = iter_through(simple_data_iter, count)
    if simple_data_iter2 is not None:
        iter_through(simple_data_iter2, count)
    
    # Closing
    pbar.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
