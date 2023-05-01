"""Calculate generation metrics
"""
import json
import os

import click
import cv2
import numpy as np
import torch
import torch_fidelity
import tqdm

import dnnlib
import legacy
from diffusions.contruct_trainer import construct_imagen_trainer
from diffusions.decode import decode_nc
from utils.simple_dataset import SimpleDatasetForMetric

torch.manual_seed(0)
np.random.seed(0)


METRIC_ROOT = 'metrics_cache'
os.makedirs(METRIC_ROOT, exist_ok=True)

@click.command()
@click.option('--real_data', type=str, required=True)
@click.option('--network_ae', 'network_ae_pkl',
              help='Network pickle filename of autoencoder.', required=True)
@click.option('--network_diff', 'network_diff_pkl',
              help='Network pickle filename of diffusion unet.', required=True)
@click.option('--diff_config', type=str, required=True)
@click.option('--exp_name', type=str, required=True)
@click.option('--generate_batch_size', type=int, default=32, show_default=True)
@click.option('--sample_total_img', type=int, default=50000, show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    device = torch.device('cuda')

    # Extract names
    data_name = os.path.basename(opts.real_data).rstrip('.zip')
    exp_name = opts.exp_name
    exported_out = os.path.join(METRIC_ROOT, exp_name)

    # Make folders
    os.makedirs(exported_out, exist_ok=True)

    # Exported Images
    with dnnlib.util.open_url(opts.network_ae_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = G.eval()
    with open(opts.diff_config, 'r') as f:
        cfg = dnnlib.EasyDict(json.load(f))
        diff_model = construct_imagen_trainer(G,
                                              cfg,
                                              device,
                                              opts.network_diff_pkl).eval()

    # Diffusion Generate images.
    g_batch_size = opts.generate_batch_size
    pbar = tqdm.tqdm(total = opts.sample_total_img // g_batch_size,
                     desc='Diffusion Generation: ')
    for i in range(opts.sample_total_img // g_batch_size):
        with torch.no_grad():
            sample_ni = diff_model.sample(batch_size=g_batch_size,
                                          use_tqdm=False)
        with torch.no_grad():
            sample_imgs = decode_nc(G, sample_ni).cpu()
            sample_imgs = sample_imgs.permute(0, 2, 3, 1)
            sample_imgs = sample_imgs.numpy()
            sample_imgs = (np.clip(sample_imgs, -1, 1) + 1) / 2.0 * 255.0
            sample_imgs = sample_imgs.astype(np.uint8)
        for j, img in enumerate(sample_imgs):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(exported_out,
                                     f'{i * g_batch_size + j:07d}.jpg'),
                        img)
        pbar.update(1)
    pbar.close()

    # Calculate
    metric_dict = torch_fidelity.calculate_metrics(
        input1=exported_out,
        input2=SimpleDatasetForMetric(opts.real_data, device=device),
        fid=True,
        isc=True,
        cache_root=METRIC_ROOT,
        cache=True,
        input2_cache_name=f'{data_name}_stats',
        cuda=True,
        verbose=True
    )

    # Results
    with open(os.path.join(METRIC_ROOT, f'{exp_name}_metric_result.json'),
              'r') as f: 
        json.dump(metric_dict, f)


if __name__ == '__main__':
    main()