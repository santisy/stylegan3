"""Add more noise scheduler to the original DDPMScheduler in difffuser"""
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DDPMScheduler
from diffusers.configuration_utils import register_to_config

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class DDPMSchedulerCustom(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "cosine_variant_v2":
            t = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float32)
            s = 0.2
            e = 1.0
            tau = 1.5
            v_start = math.cos(s * math.pi / 2.0) ** (2 * tau)
            v_end = math.cos(e * math.pi / 2.0) ** (2 * tau)
            output = torch.cos((t * (e - s) + s) * math.pi / 2) ** (2 * tau)
            output = (v_end - output) / (v_end - v_start)
            output = torch.clip(output, 1e-9, 9.999e-1)
            logsnr = torch.log(output / (1 - output))
            alphas_cumprod, _ = log_snr_to_alpha_sigma(logsnr)
            alphas_cumprod = alphas_cumprod.pow_(2.0)
            alphas_cumprod = torch.clip(alphas_cumprod, 1e-6, 0.9999)
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = 1 - alphas

        elif beta_schedule == "chen_linear":
            # This is from the paper https://arxiv.org/abs/2301.10972
            t = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = 1 - t
            alphas_cumprod = torch.clip(alphas_cumprod, 1e-6, 0.9999)
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = 1 - alphas
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type
