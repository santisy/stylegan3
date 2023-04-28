# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Tuple, Union

import click
import dnnlib
import numpy as np
import torch
import tqdm

import legacy
from utils.simple_dataset import SimpleDataset

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

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
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
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
                mu, log_var = G.img_encoder(img)
            # Check the shape
            if i == 0:
                print('NC shape in this run is', mu.shape)
            # Dump the data to folder
            mu = mu.cpu().numpy()
            log_var = log_var.cpu().numpy()
            out_file = os.path.join(outdir, f'{count:07d}.npz')
            with open(out_file, 'wb') as f:
                np.savez(f, mu=mu, log_var=log_var)
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
