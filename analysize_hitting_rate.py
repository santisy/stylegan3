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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm

import dnnlib
import legacy
from utils.simple_dataset import SimpleDataset


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the neural coordinates', type=str, required=True, metavar='DIR')
@click.option('--dataset', help='Dataset zip file.', type=str, required=True)
def generate_images(
    network_pkl: str,
    outdir: str,
    dataset: str,
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

    # Extract parameters and pre-computation
    res_max = 64
    coord_slot_stats = np.zeros((G.feat_coord_dim, res_max + 1)).astype(np.float32)

    os.makedirs(outdir, exist_ok=True)
    print(f'\033[92mSave to folder {outdir}\033[00m')

    # Construct the dataset
    simple_data_iter = iter(SimpleDataset(dataset, device))

    # Pbar and count
    pbar = tqdm.tqdm(total=len(simple_data_iter))

    def iter_through(data_iter):
        # Iter through:
        for i, img in enumerate(data_iter):
            # Neural Coordinates Output
            with torch.no_grad():
                ni = G.img_encoder(img)[0]
            if i == 0:
                print('ni shape in this run is', ni.shape)
            ni = torch.clip(ni, 0, 1)
            coord = torch.floor(ni * res_max).long().cpu().numpy()
            coord = np.clip(coord, 0, 63)
            coord2 = coord + 1
            for j, (c1, c2) in enumerate(zip(coord, coord2)):
                coord_slot_stats[j, c1.flatten()] += 1
                coord_slot_stats[j, c2.flatten()] += 1

            # Check the shape
            pbar.update(1)

    # Iter through first dataset
    iter_through(simple_data_iter)

    # Closing
    pbar.close()

    # Plotting
    coord_slot_stats = coord_slot_stats.mean(axis=0)
    coord_slot_stats = coord_slot_stats / np.sum(coord_slot_stats)
    with open(os.path.join(outdir, 'coord_split_stats.npy'), 'wb') as f:
        np.save(f, coord_slot_stats)
    sns.set_style("whitegrid")
    sns.barplot(x=np.arange(len(coord_slot_stats)),
                y=coord_slot_stats,
                palette="rocket")    
    plt.title("Indices Hitting Ratio")
    plt.xlabel("Indices")
    plt.ylabel("Ratio (%)")

    # Show the plot
    plt.show()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
