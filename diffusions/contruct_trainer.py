"""Recover diffusion trainer
"""
import torch.nn as nn

from imagen_pytorch import Unet as Unet_Imagen
from diffusions.imagen_3d import Unet3D as Unet3D_imagen
from imagen_pytorch import ImagenTrainer
from diffusions.imagen_custom import Imagen
from denoising_diffusion_pytorch import Unet as Unet_DDPM
from denoising_diffusion_pytorch import GaussianDiffusion
from diffusions.ddpm_trainer import Trainer

__all__ = ['construct_imagen_trainer']

def get_layer_attns(atten_range, total_len):
    layer_attns = []

    for i in range(1, total_len+1):
        if i in atten_range:
            layer_attns.append(True)
        else:
            layer_attns.append(False)

    return tuple(layer_attns)

def construct_imagen_trainer(G,
                             cfg,
                             device=None,
                             flag_3d=False,
                             ckpt_path=None,
                             test_flag=False):
    dim_mults = cfg.get('dim_mults', (1, 2, 2, 4))
    use_ddpm = cfg.get('use_ddpm', False)

    if not use_ddpm:
        class_embed_dim = 512
        if flag_3d:
            # This is the pure 3D convolution
            # Using the fully 3D convolutions
            unet = Unet3D_imagen(
                dim=cfg.dim,
                dim_mults=dim_mults,
            )
        else:
            unet = Unet_Imagen(dim=cfg.dim,
                               text_embed_dim=class_embed_dim,
                               channels=G.feat_coord_dim,
                               dim_mults=dim_mults,
                               num_resnet_blocks=cfg.get('num_resnet_blocks', 3),
                               layer_attns=get_layer_attns(cfg.get('atten_layers', [3, 4]),
                                                           len(dim_mults)),
                               layer_cross_attns = False,
                               use_linear_attn = True,
                               cond_on_text = cfg.class_condition)

        if cfg.class_condition:
            unet.add_module("class_embedding_layer",
                            nn.Embedding(1000, class_embed_dim))
        imagen = Imagen(
                condition_on_text = cfg.class_condition,
                unets = (unet, ),
                image_sizes = (cfg.feat_spatial_size, ),
                timesteps = 1000,
                channels=G.feat_coord_dim,
                auto_normalize_img=False,
                min_snr_gamma=5,
                min_snr_loss_weight=cfg.get('use_min_snr', True),
                dynamic_thresholding=False,
                noise_schedules=cfg.get('noise_scheduler', 'cosine'), # TODO, is this necessary? Now we are `cosine` scheduler
                pred_objectives='noise', # noise or x_start
                loss_type='l2'
                )

        cfg_precision = cfg.get("mixed_precision", "no")
        precision = None if cfg_precision == "no" else cfg_precision

        trainer = ImagenTrainer(imagen=imagen,
                                imagen_checkpoint_path=None, # TODO: continue training
                                lr=cfg.train_lr,
                                cosine_decay_max_steps=cfg.cosine_decay_max_steps,  # Note I manually change the eta_min to 1e-5
                                warmup_steps=cfg.warmup_steps,
                                use_ema=cfg.get("use_ema", True),
                                precision=precision
                                )
        if ckpt_path is not None:
            trainer.load(ckpt_path,
                        only_model=cfg.only_load_model if not test_flag else False)

    else:
        unet = Unet_DDPM(dim=cfg.dim,
                         channels=G.feat_coord_dim,
                         dim_mults=dim_mults)
        diffusion = GaussianDiffusion(unet,
                                      image_size=cfg.feat_spatial_size,
                                      num_sample_steps=1000,
                                      channels=G.feat_coord_dim,
                                      sampling_timesteps=250)
        trainer = Trainer(diffusion,
                          train_lr=cfg.train_lr,
                          ema_decay=0.995,
                          auto_normalize=True)
        if ckpt_path is not None:
            trainer.load(ckpt_path)

    if device is not None:
        trainer = trainer.to(device)

    return trainer
