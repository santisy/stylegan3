"""
    Train the diffusion on extracted key codes
    Author: Dingdong Yang
    Contact: dya62@sfu.ca
"""

import click
import json
import math
import os
import sys
import subprocess
import time

import numpy as np
import torch
import torch.distributed as tdist
sys.path.insert(0, '.')

import dnnlib
import legacy
from training.training_loop import save_image_grid
import torchvision.utils as tvu

from utils.utils import delete_file
from utils.utils import cast_device
from utils.utils import copy_back_fn
from training.dataset import ImageFolderDataset as Dataset
from diffusions.decode import decode_nc
from diffusions.contruct_trainer import construct_imagen_trainer
from imagen_pytorch import ImagenTrainer
from imagen_pytorch.trainer import cast_tuple

# Disable tqdm globally
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


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
@click.option('--dim_mults', type=lambda x:[int(y) for y in x.split(',')],
              help='The channel multiplication of the network.',
              default='1,2,2,4', show_default=True)
@click.option('--num_resnet_blocks', type=lambda x:[int(y) for y in x.split(',')],
              help='Number of residual blocks.',
              default='4,4,4,4', show_default=True)
@click.option('--noise_scheduler',
              type=click.Choice(['linear', 'cosine', 'chen_linear']),
              default='cosine')
@click.option('--no_noise_perturb', type=bool, default=True,
              help='Disable noise perturbation when tranining diffusion.')
@click.option('--resume', type=str, default=None,
              help='Resuming the training checkpoint.')
@click.option('--atten_layers', type=lambda x:[int(y) for y in x.split(',')],
              help='The resolution range to do attention.',
              default='3,4', show_default=True)
@click.option('--warmup_steps', type=int, default=None,
              show_default=True)
@click.option('--cosine_decay_max_steps', type=int, default=None,
              show_default=True)
@click.option('--only_load_model', type=bool, default=False, show_default=True)
@click.option('--use_ddpm', type=bool, default=False, show_default=True)
@click.option('--use_min_snr', type=bool, default=True, show_default=True)
@click.option('--class_condition', type=bool, default=False, show_default=True)
@click.option('--cond_scale', type=float, default=5, show_default=True)
@click.option('--work_on_tmp_dir', type=bool, default=False)
@click.option('--use_ema', type=bool, default=True, show_default=True)
@click.option('--debug', type=bool, default=False, show_default=True)
@click.option('--mixed_precision', type=str, default="no", show_default=True)
@click.option('--copy_back', type=bool, default=False, show_default=True)

def train_diffusion(**kwargs):

    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    # Prepare folder and tensorboard
    if opts.work_on_tmp_dir:
        tmp_dir = os.getenv("SLURM_TMPDIR")
    else:
        tmp_dir = ""
    local_dir = os.path.join('training_runs', opts.exp_id)
    os.makedirs(local_dir, exist_ok=True)
    run_dir = os.path.join(tmp_dir, 'training_runs', opts.exp_id)
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
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        G = G.eval()

    # Diffusion Unet Module and optimizer --------------------
    trainer = construct_imagen_trainer(G, opts, device=None, ckpt_path=opts.resume)
    G = G.to(trainer.device) # This device is coming from accelerator


    rank_now = tdist.get_rank() if tdist.is_initialized() else 0
    world_size = tdist.get_world_size() if tdist.is_initialized() else 1
    print(f"Rank now {rank_now}")
    print(f"World size is {world_size}")
    # Copy dataset if necessary
    if opts.work_on_tmp_dir:
        new_data_root = os.path.join(tmp_dir, "datasets")
        os.makedirs(new_data_root, exist_ok=True)
        dataset_path = os.path.join(new_data_root, os.path.basename(opts.dataset))
    else:
        dataset_path = opts.dataset
    if trainer.accelerator.is_local_main_process and opts.work_on_tmp_dir and not os.path.exists(dataset_path):
        print(f"\033[92mCopying dataset {opts.dataset} to {tmp_dir} ...\033[00m")
        os.system(f"cp {opts.dataset} {new_data_root}") 
        print("\033[92mFinished copying.\033[00m")
    if tdist.is_initialized():
        tdist.barrier()

    # Set randomness
    np.random.seed(rank_now)
    torch.manual_seed(rank_now)
    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Dataset constructing and preparing ------------------------------

    # Dataset, sampler and iterator
    # Add the collate_fn + encode part
    dataset = Dataset(dataset_path,
                      imagenet_flag=opts.class_condition)
    # Collate fn
    def collate_fn(batch_tuple):
        img = torch.tensor(torch.stack([torch.tensor(x[0]) for x in batch_tuple], dim=0))
        img = img.float() / 127.5 - 1.0
        if not opts.class_condition:
            label = None
        else:
            # Shift the label to +1
            # Zero is the unconditional label
            label = torch.stack([torch.tensor(x[1]) for x in batch_tuple]).flatten().long()
            # Reuse the text condition here
            label = label.unsqueeze(dim=1)
        return img, label
    # Contruct dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    # Add dataloader
    trainer.add_train_dataloader(train_dataloader)
    # Redefine the `step_with_dl_iter` function
    def step_with_dl_iter(self, dl_iter, **kwargs):
        if opts.class_condition:
            keywords_name = self.dl_tuple_output_keywords_names[:2]
        else:
            keywords_name = (self.dl_tuple_output_keywords_names[0],)
        # Cast device in compatible with the `collate_fn`
        dl_tuple_output = cast_device(cast_tuple(next(dl_iter)), self.device)
        model_input = dict(list(zip(keywords_name, dl_tuple_output)))
        with torch.no_grad():
            img_ = dl_tuple_output[0]
            encode_fn = getattr(G, "encode",
                                lambda x, **kwargs: (G.img_encoder(x),))
            try:
                encoded = encode_fn(img_,
                                        no_noise_perturb=opts.no_noise_perturb
                                        )[0].detach()
            except:
                encoded = encode_fn(img_)[0].detach()
            model_input[keywords_name[0]] = (encoded - 0.5) * 2.0
        if opts.class_condition:
            label = model_input["text_embeds"]
            model_input["text_embeds"] = self.imagen.unets[0].class_embedding_layer(label)
        loss = self.forward(**{**kwargs, **model_input})
        return loss    
    # Rebind
    trainer.step_with_dl_iter = step_with_dl_iter.__get__(trainer, ImagenTrainer)
    # --------------------------------------------------

    # Main process flag
    main_p_flag = trainer.accelerator.is_main_process

    # Counting initials
    count = 0
    iter_count = 0
    save_snapshot_list = []
    start_time = time.time()
    tick_end_time = None

    if main_p_flag:
        print("\033[92mStart training\033[00m")

    # Main Loop Starts Here --------------------
    while True:
        # The data loader now is whitin the trainer 
        loss = trainer.train_step(unet_number = 1)

        # Increase couting
        count += opts.batch_size * world_size
        iter_count += 1
        global_step = count // 1000

        # Recording
        if count % (opts.record_k * 1000) == 0 and main_p_flag:
            cur_lr = trainer.get_lr(unet_number=1)
            stats_tfevents.add_scalar('Loss/diff_loss', loss,
                                      global_step=global_step)
            stats_tfevents.add_scalar('Progress/lr', cur_lr,
                                      global_step=global_step)
            tick_end_time = time.time()
            print(f'Iters {iter_count / 1000.: .2f}k\t'
                  f'kimg {count // 1000}\t'
                  f'loss {loss:.4f}\t'
                  f'learning_rate {cur_lr: .6f}\t'
                  f'time {dnnlib.util.format_time(tick_end_time - start_time)}\t')

        # Sampling
        if count % (opts.sample_k * 1000) == 0 and main_p_flag:
            print('Save image ...')
            # Get the label
            if opts.class_condition:
                label = torch.randint(0, 1000, size=(opts.sample_num, 1),
                                      device=trainer.device).long()
                class_embeds = trainer.ema_unets[0].class_embedding_layer(label)
                cond_scale = opts.cond_scale
            else:
                class_embeds = None
                cond_scale = 1.0

            # Set random label
            sample_ni = trainer.sample(batch_size=opts.sample_num,
                                       text_embeds=class_embeds, # TODO
                                       cond_scale=cond_scale,
                                       use_tqdm=False)
            print('Value range of sampled nc: ', sample_ni.min(), sample_ni.max())
            print('Stats of sampled nc:', sample_ni.mean(), sample_ni.var())
            sample_ni = torch.clip((sample_ni + 1) / 2.0, 0, 1)
            with torch.no_grad():
                sample_imgs = decode_nc(G, sample_ni).cpu().numpy()
            # Save image to local target folder
            save_image_path = os.path.join(run_dir,
                                           f'fakes{global_step:06d}.png')
            save_image_grid(sample_imgs,
                            save_image_path,
                            drange=[-1,1],
                            grid_size=(int(np.sqrt(opts.sample_num)),
                                       int(np.sqrt(opts.sample_num))))
            if opts.copy_back:
                copy_back_fn(save_image_path, local_dir)
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
            trainer.save(save_file) # Here is a waiting for everyone!

            if opts.copy_back and main_p_flag:
                copy_back_fn(save_file, local_dir)
            if len(save_snapshot_list) > 5 and main_p_flag:
                delete_file(save_snapshot_list.pop(0))

        # Set the barrier
        if tdist.is_initialized() and (
            count % (opts.record_k * 1000) == 0 or
            count % (opts.sample_k * 1000) == 0 or
            count % (opts.snap_k * 1000) == 0
        ):
            tdist.barrier()

if __name__ == '__main__':
    train_diffusion()
