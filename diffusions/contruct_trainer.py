"""Recover diffusion trainer
"""
from imagen_pytorch import Unet, Imagen, ImagenTrainer

__all__ = ['construct_imagen_trainer']

def construct_imagen_trainer(G, cfg, device, ckpt_path=None, test_flag=False):

    unet = Unet(dim=cfg.dim,
                channels=G.feat_coord_dim,
                dim_mults=cfg.get('dim_mults', (1, 2, 2, 4)),
                num_resnet_blocks=cfg.get('num_resnet_blocks', 3),
                layer_attns=(False, False, True, True),
                layer_cross_attns = False,
                use_linear_attn = True,
                cond_on_text=False
                )

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

    return trainer.to(device)
