# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
from einops import reduce
import torch
import torch.nn.functional as F

from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.vgg_perceptual_loss import VGGPerceptualLoss
from taming.modules.losses.lpips import LPIPS


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, d_weight_ori=0.5):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * d_weight_ori
    return d_weight

def kl_loss(mu: torch.Tensor,
            log_var: torch.Tensor,
            kl_loss_weight: float=0.01,
            kl_std: float=1.0,
            ) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*kl_std )
    sampled_dist = torch.distributions.normal.Normal( mu, std )
    kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
    kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean() * kl_loss_weight

    return kl_loss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D,
                 augment_pipe=None,
                 r1_gamma=10,
                 style_mixing_prob=0,
                 pl_weight=0,
                 pl_batch_shrink=2,
                 pl_decay=0.01,
                 pl_no_weight_grad=False,
                 blur_init_sigma=0,
                 blur_fade_kimg=0,
                 encoder_flag=False,
                 l2loss_weight=20.0,
                 use_kl_reg=False,
                 kl_loss_weight=1e-4,
                 vq_decoder=False,
                 disc_start=50000,
                 flag_3d=False,
                 ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.encoder_flag       = encoder_flag
        self.l2loss_weight      = l2loss_weight
        self.use_kl_reg         = use_kl_reg
        self.kl_loss_weight     = kl_loss_weight
        self.vq_decoder         = vq_decoder
        self.disc_start         = disc_start
        self.flag_3d            = flag_3d
        if encoder_flag:
            self.vgg_perceptual_loss = VGGPerceptualLoss().to(device)
        if vq_decoder:
            self.perceptual_loss = LPIPS().eval().to(device)

    def run_G(self, z, c, real_img, update_emas=False):
        ws1, ws2 = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        if not self.encoder_flag:
            img = self.G.synthesis(ws1, ws2, z, update_emas=update_emas)
            mu, log_var = None, None
        else:
            out = self.G.synthesis(ws1, ws2, real_img, update_emas=update_emas, return_kl_terms=self.use_kl_reg)
            if not self.use_kl_reg:
                img = out
                mu, log_var = None, None
            else:
                img, mu, log_var = out
        return img, (mu, log_var)

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain,
                             cur_nimg, cur_tick):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, kl_term = self.run_G(gen_z, gen_c, real_img)
                loss_Gmain = 0

                if self.encoder_flag:
                    if not self.vq_decoder:
                        # L2 Loss
                        loss_l2 = F.mse_loss(gen_img, real_img) * self.l2loss_weight
                        loss_Gmain += loss_l2
                        training_stats.report('Loss/G/l2loss', loss_l2)
                        if not self.flag_3d:
                            # VGG loss
                            loss_vgg = self.vgg_perceptual_loss(gen_img, real_img) * 5.0
                            loss_Gmain += loss_vgg
                            training_stats.report('Loss/G/vggloss', loss_vgg)
                    else:
                        rec_loss = torch.abs(real_img.contiguous() - gen_img.contiguous()).mean()
                        loss_percep = self.perceptual_loss(
                            ((real_img + 1) / 2.0).contiguous(),
                            ((gen_img + 1) / 2.0).contiguous()).mean()
                        loss_Gmain += (loss_percep + rec_loss)
                        training_stats.report('Loss/G/loss_precep', loss_percep)
                    # KL loss
                    if self.use_kl_reg:
                        loss_kl = kl_loss(kl_term[0], kl_term[1],
                                          kl_loss_weight=self.kl_loss_weight,
                                          kl_std=1.0)
                        loss_Gmain += loss_kl
                        training_stats.report('Loss/G/klloss', loss_kl)

                if not self.flag_3d:
                    if not self.vq_decoder:
                        gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                        training_stats.report('Loss/scores/fake', gen_logits)
                        training_stats.report('Loss/signs/fake', gen_logits.sign())
                        g_loss = torch.nn.functional.softplus(-gen_logits).mean() # -log(sigmoid(gen_logits))
                        loss_Gmain += g_loss
                    else:
                        gen_logits = self.D(gen_img.contiguous())
                        g_loss = -torch.mean(gen_logits)
                        g_weight = calculate_adaptive_weight(torch.mean(loss_percep),
                                                            g_loss,
                                                            self.G.synthesis_network.conv_out.weight)
                        if cur_tick < self.disc_start:
                            g_weight = 0
                        loss_Gmain += g_loss * g_weight
                        training_stats.report('Loss/G/g_weight', g_weight)

                    training_stats.report('Loss/G/loss', g_loss)
                

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_img, update_emas=True)
                if not self.vq_decoder:
                    gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                else:
                    gen_logits = self.D(gen_img.contiguous().detach())
                    real_logits = self.D(real_img.contiguous().detach())
                    if cur_tick < self.disc_start:
                        dis_factor = 0
                    else:
                        dis_factor = 1.0
                    loss_Dgen = hinge_d_loss(real_logits, gen_logits) * dis_factor
                        
                    training_stats.report('Loss/D/loss', loss_Dgen)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth'] and not self.vq_decoder:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
