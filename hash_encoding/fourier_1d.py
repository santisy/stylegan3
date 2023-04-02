"""
Author: Dingdong Yang
Start Date: 04/01/2023
Reference Repo: https://github.com/neuraloperator/neuraloperator
"""
from functools import partial

import torch
import torch.nn as nn

from training.networks_stylegan2 import FullyConnectedLayer


__all__ = ['SpectralConv1d']


#Complex multiplication
def modulated_compl_mul1d(a, b, s, decoefs):
    """a is the complex values"""
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    a = torch.view_as_real(a)
    op = partial(torch.einsum, "bix,iox->box")
    out = torch.stack([
        op(a[..., 0] * s[..., 0], b[..., 0]) * decoefs[..., 0] - op(a[..., 1] * s[..., 1], b[..., 1]) * decoefs[..., 1],
        op(a[..., 1] * s[..., 0], b[..., 0]) * decoefs[..., 0] + op(a[..., 0] * s[..., 1], b[..., 1]) * decoefs[..., 1]
    ], dim=-1)

    return torch.view_as_complex(out)

class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 s_dim: int,
                 modes1: int=16,
                 ):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        Args:
            in_channels:
            out_channels:
            s_dim: style dimension
            modes1 (int): how many top frequency modes to be kept. 
                (default: 16)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))
        self.s_mapping = FullyConnectedLayer(s_dim, in_channels*2, bias_init=1)

    def _get_decoefs(self, s):
        b = s.size(0)
        w = self.weights1
        w = w.unsqueeze(dim=0) # 1 x in x out x m x 2
        s = s.reshape(b, self.in_channels, 1, 1, 2)
        w = w * s
        decoefs = (w.square().sum(dim=[1, 3]) + 1e-8).rsqrt() # B x O x 2
        decoefs = decoefs.unsqueeze(dim=2)

        return decoefs

    def forward(self, x, s):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # Modulation
        s = self.s_mapping(s)
        decoefs = self._get_decoefs(s) 
        s = s.reshape(batchsize, self.in_channels, 1, 2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = modulated_compl_mul1d(x_ft[:, :, :self.modes1],
                                                           self.weights1,
                                                           s, decoefs)

        #Return to physical space
        x = torch.fft.irfft(out_ft, dim=-1, norm='ortho')
        return x
    
if __name__ == '__main__':
    sp_layer = SpectralConv1d(16, 16, s_dim=128, modes1=16)
    input_var = torch.randn(4, 16, 128)
    s = torch.randn(4, 128)
    output_var = sp_layer(input_var, s)
    print(output_var.shape)
    