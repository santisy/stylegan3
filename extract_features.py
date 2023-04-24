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
from typing import List, Optional, Tuple, Union
import zipfile

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import tqdm

import legacy

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
class SimpleDataset:
    def __init__(self, dataset: str):
        """
            Simple dataset to extract the images from zip file and return a 
            batch size 1 CUDA tensor.
        """

        assert os.path.isfile(dataset) and dataset.endswith('zip')
        self._zipfile = zipfile.ZipFile(dataset)
        self._all_fnames = set(self._zipfile.namelist())
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames
                                    if self._file_ext(fname)
                                    in PIL.Image.EXTENSION)

        self.data_len = len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        return self._zipfile.open(fname, 'r')

    def __len__(self):
        return self.data_len

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.data_len:
            fname = self._image_fnames[self.n]
            self.n += 1
            with self._open_file(fname) as f:
                image = np.array(PIL.Image.open(f))
            image = image.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(image.copy()).float()
            img_tensor = img_tensor / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(dim=0).cuda()
            return img_tensor
        else:
            raise StopIteration



#----------------------------------------------------------------------------

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

    os.makedirs(outdir, exist_ok=True)
    print(f'\033[92mExtract to folder {outdir}\033[00m')

    # Construct the dataset
    simple_data_iter = iter(SimpleDataset(dataset))

    # Pbar
    pbar = tqdm.tqdm(total=len(simple_data_iter))

    # Iter through:
    for i, img in enumerate(simple_data_iter):
        # Neural Coordinates Output
        nc_out = G.img_encoder(img)
        # Check the shape
        if i == 0:
            print('NC shape in this run is', nc_out.shape)
        # Dump the data to folder
        nc_out = nc_out.cpu().numpy()
        out_file = os.path.join(outdir, f'{i:07d}.npy')
        with open(out_file, 'wb') as f:
            np.save(f, nc_out)
        pbar.update(1)
    
    # Closing
    pbar.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
