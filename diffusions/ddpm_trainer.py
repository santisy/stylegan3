import math

import torch
from ema_pytorch import EMA
from torch.optim import Adam
from accelerate import Accelerator
from denoising_diffusion_pytorch.version import __version__

__all__ = ['Trainer']

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def exists(x):
    return x is not None

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        train_lr = 1e-4,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        amp = False,
        fp16 = False,
        split_batches = True,
        auto_normalize = True
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.image_size = diffusion_model.image_size

        # optimizer
        self.train_lr = train_lr
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.auto_normalize = auto_normalize

    @property
    def device(self):
        return self.accelerator.device

    def save(self, save_path):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, save_path)

    def load(self, load_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(load_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def sample(self, batch_size: int):
        with torch.no_grad:
            self.ema.ema_model.eval()
            out = self.ema.ema_model.sample(batch_size=batch_size)
        if self.auto_normalize:
            return (out + 1.0) / 2.0

        return out

    def update(*args, **kwargs):
        """Place holder"""
        pass

    def get_lr(self, *args, **kwargs):
        return self.train_lr

    def __call__(self, data):
        if self.auto_normalize:
            data = data * 2.0 - 1.0

        accelerator = self.accelerator
        device = accelerator.device
        total_loss = 0.

        for _ in range(self.gradient_accumulate_every):

            with self.accelerator.autocast():
                loss = self.model(data)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

            self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()
        self.ema.update()

        self.step += 1

