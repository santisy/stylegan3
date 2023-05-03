"""Find the nearest neighbor generation
"""
import glob
import os
import shutil

import click
import cv2
import lpips
from PIL import Image
import numpy as np
import tqdm
import torch

import dnnlib
from utils.simple_dataset import SimpleDataset


@click.command()
@click.option('--dataset', help='Dataset zip file.', type=str, required=True)
@click.option('--indir',
              help=('Find the nearest images in training datasets under this'
                    ' folder.'),
              type=str, required=True)
@click.option('--outdir', help='Where to save the images', type=str,
              required=True)
def find_the_nearest(**kwargs):
    
    # The device
    device = torch.device('cuda')

    # Parse to a easy dictionary
    opts = dnnlib.EasyDict(kwargs)

    # Find images
    input_img_paths = sorted(glob.glob(os.path.join(opts.indir, "*.png")))

    # LPIPS fn
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Construct the training dataset
    dataset = SimpleDataset(opts.dataset, device)
    data_iter = iter(dataset)

    for input_img_path in input_img_paths:
        # Prepapre output folder
        img_name = os.path.basename(input_img_path).split('.')[0]
        outdir_now = os.path.join(opts.outdir, img_name)
        os.makedirs(outdir_now, exist_ok=True)

        # Prepare input generated image
        img = cv2.imread(input_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(dim=0)
        img = img / 127.5 - 1.0
        img = img.to(device)

        loss_collect = np.zeros((len(data_iter)))

        pbar = tqdm.tqdm(total=len(data_iter),
                         desc=f'Finding nearest agains image {img_name}')
        # Iters all the training data
        for i, real_img in enumerate(data_iter):
            loss_collect[i] = loss_fn(real_img, img)
            pbar.update(1)
        pbar.close()

        # Find the 10 smallest and save the image
        shutil.copy(input_img_path, os.path.join(outdir_now, 'input.png'))
        smallest_10_indices = np.argsort(loss_collect)[:10]
        for order, idx in enumerate(smallest_10_indices):
            Image.fromarray(data_iter.retrieve_by_idx(idx), 'RGB').save(
                os.path.join(outdir_now, f'order{order}_{idx:08d}.png'))

        # Reset the iterator
        data_iter = iter(dataset)



if __name__ == "__main__":
    find_the_nearest()