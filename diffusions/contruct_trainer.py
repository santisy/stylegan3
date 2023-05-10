"""Recover diffusion trainer
"""
from imagen_pytorch import Unet as Unet_Imagen
from imagen_pytorch import Imagen, ImagenTrainer
from denoising_diffusion_pytorch import Unet as Unet_DDPM
from denoising_diffusion_pytorch import ContinuousTimeGaussianDiffusion
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

def construct_imagen_trainer(G, cfg, device, ckpt_path=None, test_flag=False):
    dim_mults = cfg.get('dim_mults', (1, 2, 2, 4))
    use_ddpm = cfg.get('use_ddpm', False)

    if not use_ddpm:
        unet = Unet_Imagen(dim=cfg.dim,
                           channels=G.feat_coord_dim,
                           dim_mults=dim_mults,
                           num_resnet_blocks=cfg.get('num_resnet_blocks', 3),
                           layer_attns=get_layer_attns(cfg.get('atten_layers', [3, 4]),
                                                       len(dim_mults)),
                           layer_cross_attns = False,
                           use_linear_attn = True,
                           cond_on_text=False)

        imagen = Imagen(
                condition_on_text = False,
                unets = (unet, ),
                image_sizes = (cfg.feat_spatial_size, ),
                timesteps = 1000,
                channels=G.feat_coord_dim,
                auto_normalize_img=True,
                min_snr_gamma=5,
                min_snr_loss_weight=True,
                dynamic_thresholding=False,
                noise_schedules=cfg.get('noise_scheduler', 'cosine'), # TODO, is this necessary? Now we are `cosine` scheduler
                pred_objectives='noise', # noise or x_start
                loss_type='l2'
                )

        trainer = ImagenTrainer(imagen=imagen,
                                imagen_checkpoint_path=None, # TODO: continue training
                                lr=cfg.train_lr,
                                cosine_decay_max_steps=cfg.cosine_decay_max_steps,  # Note I manually change the eta_min to 1e-5
                                warmup_steps=cfg.warmup_steps
                                )
        if ckpt_path is not None:
            trainer.load(ckpt_path,
                        only_model=cfg.only_load_model if not test_flag else False)

    else:
        unet = Unet_DDPM(dim=cfg.dim,
                         channels=G.feat_coord_dim,
                         dim_mults=dim_mults)
        diffusion = ContinuousTimeGaussianDiffusion(unet,
                                                    image_size=cfg.feat_spatial_size,
                                                    num_sample_steps=1000,
                                                    channels=G.feat_coord_dim,
                                                    loss_type='l1')
        trainer =  Trainer(diffusion,
                           train_lr=cfg.train_lr,
                           auto_normalize=True)
        trainer.load(ckpt_path)


    return trainer.to(device)
