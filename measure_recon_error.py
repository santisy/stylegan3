"""Reconstruct and measure the metric on validation dataset.
"""
import os

import click
import cv2
import numpy as np
import tqdm
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure

import dnnlib
import legacy
import lpips
from utils.simple_dataset import SimpleDataset



@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename',
              required=True)
@click.option('--dataset', help='Dataset zip file.', type=str, required=True)
@click.option('--outdir', help='Where to save the images', type=str,
              required=True)
@click.option('--last_split_n',
              help='How many last files to be measured if provided',
              type=int, default=-1, show_default=True)
@click.option('--runs', help='How many runs in total', type=int, default=1,
              show_default=True)
@click.option('--max_save',
              help='Maximum number of reconstructed images to be saved.',
              default=-1, show_default=True, type=int)
def test_recon_main(
    network_pkl: str,
    dataset: str,
    outdir: str,
    last_split_n: int=-1,
    runs: int=10,
    max_save: int=-1,
):

    # The device
    device = torch.device('cuda')

    # Load the network
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = G.eval()

    # Make the folder
    os.makedirs(outdir, exist_ok=True)
    f = open(os.path.join(outdir, 'metric.txt'), 'a')
    print(f'\033[92mSave reconstructed to folder {outdir}\033[00m')

    # Construct dataset
    data_iter = iter(SimpleDataset(dataset, device,
                                    last_split_n=last_split_n))
    
    # Construct LPIPS forward instance
    loss_fn = lpips.LPIPS(net='alex').to(device)
    psnr_fn = PeakSignalNoiseRatio().to(device)
    ssim_fn = StructuralSimilarityIndexMeasure().to(device)

    # Loss collection
    lpips_collect = torch.zeros(runs, len(data_iter)).to(device)
    psnr_collect = torch.zeros(runs, len(data_iter)).to(device)
    ssim_collect = torch.zeros(runs, len(data_iter)).to(device)

    # Iterate
    count = 0
    pbar = tqdm.tqdm(total=len(data_iter) * runs)

    for j in range(runs):
        for i, img in enumerate(data_iter):
            # Test forward
            with torch.no_grad():
                rec = G(img)

            # Collect metric
            lpips_collect[j, i] = loss_fn(img, rec).detach()
            psnr_collect[j, i] = psnr_fn(rec, img).detach()
            ssim_collect[j, i] = ssim_fn(rec, img).aetach()

            # Save images (only save images at the first run)
            if j == 0 and ((max_save > 0 and count < max_save) or max_save < 0):
                rec_img = rec[0].permute(1, 2, 0).cpu()
                rec_img = torch.clip(rec_img, -1, 1)
                rec_img = (rec_img.numpy() + 1.0) / 2.0 * 255.0
                rec_img = cv2.cvtColor(rec_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(outdir, f'rec_{i:06d}.png'), rec_img)
            
            # Update pbar and counts
            pbar.update(1)
            count += 1

    # Record LPIPS
    lpips_collect = lpips_collect.mean(dim=0)
    f.write(f'LPIPS: \nmean {lpips_collect.mean().item(): .4f} \t'
            f' 2*std {lpips_collect.std().item() * 2: .6f}\n')
    # Record PSNR
    psnr_collect = psnr_collect.mean(dim=0)
    f.write(f'PSNR: \nmean {psnr_collect.mean().item(): .4f} \t'
            f' 2*std {psnr_collect.std().item() * 2: .6f}\n')
    # Record SSIM
    ssim_collect = ssim_collect.mean(dim=0)
    f.write(f'SSIM: \nmean {ssim_collect.mean().item(): .4f} \t'
            f' 2*std {ssim_collect.std().item() * 2: .6f}\n')

    f.close()
    pbar.close()


if __name__ == '__main__':
    test_recon_main()
