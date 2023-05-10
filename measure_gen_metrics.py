"""Calculate generation metrics
"""
import json
import os

import click
import cv2
import glob
import numpy as np
import torch
import torch_fidelity
import tqdm

import dnnlib
import legacy
from diffusions.contruct_trainer import construct_imagen_trainer
from diffusions.decode import decode_nc
from diffusions.dpm_solver import DPM_Solver 
from diffusions.dpm_solver import NoiseScheduleVP 
from diffusions.dpm_solver import GaussianDiffusionContinuousTimes
from diffusions.dpm_solver import log_snr_to_alpha_sigma

torch.manual_seed(0)
np.random.seed(0)


METRIC_ROOT = 'metrics_cache'
os.makedirs(METRIC_ROOT, exist_ok=True)

def _measure_and_save(out_dir: str,
                      real_dir: str,
                      data_name: str,
                      exp_name: str,
                      eval_num: int):
    print(f'\033[093mEvaluation on {eval_num/1000:.2f}k image.\033[00m')
    # Calculate
    metric_dict = torch_fidelity.calculate_metrics(
        input1=out_dir,
        input2=real_dir,
        fid=True,
        isc=True,
        cache_root=METRIC_ROOT,
        cache=True,
        input2_cache_name=f'{data_name}_stats',
        cuda=True,
        verbose=True,
        samples_find_deep=True
    )

    # Results
    with open(os.path.join(METRIC_ROOT,
                           f'{exp_name}_metric_result.txt'),
              'a') as f: 
        f.write(f'{eval_num}k img evaluation results:\n')
        json.dump(metric_dict, f, ensure_ascii=False)
        f.write('\n')


@click.command()
@click.option('--real_data', type=str, required=True)
@click.option('--network_ae', 'network_ae_pkl', type=str,
              help='Network pickle filename of autoencoder.',
              default=None)
@click.option('--network_diff', 'network_diff_pkl', type=str,
              help='Network pickle filename of diffusion unet.',
              default=None)
@click.option('--diff_config', type=str, default=None)
@click.option('--input_folder', type=str,
              help='If given, will evaluate the images here.',
              default=None)
@click.option('--every_k', type=int,
              help='If given, it will calculate fid and more every k images.',
              default=None)
@click.option('--exp_name', type=str, required=True)
@click.option('--generate_batch_size', type=int, default=32, show_default=True)
@click.option('--sample_total_img', type=int, default=50000, show_default=True)
@click.option('--skip_gen', type=bool, default=False, show_default=True,
              help='Skip the generation process.')
@click.option('--use_dpm_solver', type=bool,
              help='Use DPM solver or not to accelerate the sampling.',
              default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    device = torch.device('cuda')

    # Extract names and variables
    data_name = os.path.basename(opts.real_data).rstrip('.zip')
    exp_name = opts.exp_name
    g_batch_size = opts.generate_batch_size

    # Construct networks
    if opts.input_folder is None:
        assert (opts.network_ae_pkl is not None and
                opts.network_diff_pkl is not None and
                opts.diff_config is not None)
        exported_out = os.path.join(METRIC_ROOT, exp_name)
        # Exported Images
        with dnnlib.util.open_url(opts.network_ae_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
            G = G.eval()
        with open(opts.diff_config, 'r') as f:
            cfg = dnnlib.EasyDict(json.load(f))
            diff_model = construct_imagen_trainer(G,
                                                cfg,
                                                device,
                                                opts.network_diff_pkl,
                                                test_flag=True).eval()
    else:
        exported_out = opts.input_folder

    # Make folders
    os.makedirs(exported_out, exist_ok=True)


    # Wrap it to DPM-solver
    if opts.use_dpm_solver:
        noise_scheduler = GaussianDiffusionContinuousTimes(
            noise_schedule=cfg.get('noise_scheduler', 'cosine'),
            timesteps=1000)
        t = torch.linspace(0, 1, 1000, device=device)
        log_snr = noise_scheduler.log_snr(t)
        alphas, _ = log_snr_to_alpha_sigma(log_snr)
        alphas_cumprod = (alphas * alphas).detach()
        dpm_solver = DPM_Solver(diff_model.unets[0],
                                NoiseScheduleVP(alphas_cumprod=alphas_cumprod),
                                algorithm_type='dpmsolver')
    

    if not opts.skip_gen and opts.input_folder is None:
        # Diffusion Generate images.
        pbar = tqdm.tqdm(total = opts.sample_total_img // g_batch_size + 1,
                        desc='Diffusion Generation: ')
        measure_k_count = 0
        for i in range(opts.sample_total_img // g_batch_size + 1):
            with torch.no_grad():
                if opts.use_dpm_solver:
                    sample_ni = dpm_solver.sample(
                                        torch.randn(g_batch_size,
                                                    G.feat_coord_dim,
                                                    cfg.feat_spatial_size,
                                                    cfg.feat_spatial_size).to(device),
                                        steps=100,
                                        order=3,
                                        skip_type="time_uniform",
                                        method="multistep",
                                    )
                    sample_ni = (sample_ni + 1.0) / 2.0
                    print(sample_ni.max(), sample_ni.min())
                    sample_ni = torch.clip(sample_ni, 0, 1)
                    import pdb; pdb.set_trace()
                else:
                    sample_ni = diff_model.sample(batch_size=g_batch_size,
                                                  use_tqdm=False)
                    sample_ni = torch.clip(sample_ni, 0, 1)
                if i == 0:
                    print(f'\033[92mThe sample result size is {sample_ni.shape}.\033[00m')
            with torch.no_grad():
                sample_imgs = decode_nc(G, sample_ni).cpu()
                sample_imgs = sample_imgs.permute(0, 2, 3, 1)
                sample_imgs = sample_imgs.numpy()
                sample_imgs = (np.clip(sample_imgs, -1, 1) + 1) / 2.0 * 255.0
                sample_imgs = sample_imgs.astype(np.uint8)
            for j, img in enumerate(sample_imgs):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(exported_out,
                                        f'{i * g_batch_size + j:07d}.png'),
                            img)
            pbar.update(1)

            if (opts.every_k is not None and
                (i + 1) * g_batch_size // (opts.every_k * 1000) > measure_k_count) or \
                i == opts.sample_total_img // g_batch_size:
                measure_k_count += 1
                _measure_and_save(exported_out,
                                  opts.real_data,
                                  data_name,
                                  exp_name,
                                  (i + 1) * g_batch_size)


        pbar.close()
    else:
        total_num = len(glob.glob(os.path.join(exported_out, "**", "*.png"),
                                  recursive=True))
        print('\033[93m[WARNING] Skipping the generation process.\033[00m')

        _measure_and_save(exported_out,
                          opts.real_data,
                          data_name,
                          exp_name,
                          total_num)

if __name__ == '__main__':
    main()
