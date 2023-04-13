"""
    Other networks
"""
import numpy as np

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.networks_stylegan2 import FullyConnectedLayer
from training.networks_stylegan2 import SynthesisLayer


flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class MappingNetwork(nn.Module):
    def __init__(self,
                 in_ch: int,
                 map_depth: int,
                 style_dim: int,
                 hidden_ch: int=None,
                 use_layer_norm: bool=False,
                 two_style_code: bool=False,
                 split_depth: int=2,
                 activation: nn.Module=nn.LeakyReLU):
        """
            Args:
                in_ch (int): input channel size. Usually is the z-size.
                map_depth (int): Map detph of the network, namely how many
                    layers.
                style_dim (int): style dimension of the code.
                hidden_ch (int): hidden channel of the mapping network.
                use_layer_norm (bool): Use layer normamlization or not.
                    (default: False)
                two_style_code (bool): Output two style codes or not.
                split_depth (int): If two style code would be output, what is
                    the depth of each MLPs. (default: 4)
                activation (nn.Module): Activation function
        """
        super().__init__()
        self.two_style_code = two_style_code

        if use_layer_norm:
            layer_norm = nn.LayerNorm
        else:
            layer_norm = nn.Identity
        linear_layer = lambda in_ch, out_ch: [FullyConnectedLayer(
                                                in_ch, out_ch, activation='lrelu'),
                                              layer_norm(out_ch)]

        # Main mapping path
        s_mapping = []
        s_mapping.extend(linear_layer(in_ch, hidden_ch))
        for _ in range(map_depth-2):
            s_mapping.extend(linear_layer(hidden_ch, hidden_ch))
        s_mapping.extend(linear_layer(hidden_ch, style_dim)) 
        self.main_path = nn.Sequential(*s_mapping)

        # Split two style-code
        if two_style_code:
            self.s1_branch = nn.Sequential(*flatten([linear_layer(style_dim, style_dim) 
                                             for _ in range(split_depth)]))
            self.s2_branch = nn.Sequential(*flatten([linear_layer(style_dim, style_dim)
                                             for _ in range(split_depth)]))

    def forward(self, z: torch.Tensor):
        z = normalize_2nd_moment(z.to(torch.float32))
        s = self.main_path(z)
        if self.two_style_code:
            s1 = self.s1_branch(s)
            s2 = self.s2_branch(s)
        else:
            s1 = s
            s2 = s

        return s1, s2

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b n t o -> b n o t')
        x = self.norm(x)
        return einops.rearrange(x, 'b n o t -> b n t o')

class MultiHeadOffsetNetwork(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 token_num: int,
                 deformable_head_num: int,
                 activation: nn.Module=nn.GELU,
                 scale: int=None):
        """
            in_dim (int): input dimension of the token
            token_num (int): table number (token number).
            deformable_head_num (int): deformable head number
            activation (nn.Module): activation function
            scale: Control the offset scale.
                #NOTE: to be determined if this is necessary
        """
        super().__init__()
        self.deformable_head_num = deformable_head_num
        self.in_dim = in_dim
        self.out_dim = out_dim

        weight = torch.randn(deformable_head_num, token_num, out_dim, in_dim)
        weight.normal_(std=np.sqrt(6.0 / (out_dim + in_dim)))
        self.register_parameter('weight', nn.Parameter(weight))
        bias = torch.zeros(1, deformable_head_num, token_num, 1)
        self.register_parameter('bias', nn.Parameter(bias))

        self.activ = activation()

        self.linear_layer = nn.Linear(out_dim, out_dim, bias=False)
        self.proxy_layer_norm = LayerNormProxy(token_num)

        # Original coordinate grids
        x = torch.linspace(-1, 1, out_dim)
        y = torch.linspace(-1, 1, token_num)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ori_grids = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(dim=0)
        self.register_buffer('ori_grids', ori_grids)

    def forward(self, x: torch.Tensor):
        """
            Args:
                x: input token B x T x C
        """
        b, t, _ = x.shape

        # B x N x T x O
        x = torch.einsum('ntoc,btc->bnto', self.weight, x) + self.bias
        x = self.proxy_layer_norm(x)
        x = self.activ(x)
        x = self.linear_layer(x)
        x = F.tanh(x)
        # For x-axis/along table, the offsets should be arbitrary,
        #   but should be close to exact entry position, avoiding bilinear
        #   interpolation.
        # x = smooth_ceiling(x * (self.out_dim // 2)) / float(self.out_dim // 2)
        x = x.reshape(b, self.deformable_head_num * t, self.out_dim, 1)
        # For y-axis/cross table, the offset is zero,
        #   namely only do within table offsets
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
        x = x + self.ori_grids.repeat(1, self.deformable_head_num, 1, 1)
        x = torch.clip(x, -1, 1)
        return x # B x (DH_NUM x T) x C x 2


class MiniLinearDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(32, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 3),
                                     nn.Tanh())
    
    def forward(self, x):
        y = self.linear1(x)
        return y


class SynthesisStack(nn.Module):
    def __init__(self, ch: int, style_dim: int, resolution: int,
                 layer_num: int=2,
                 out_ch: int=3,
                 ):

        super().__init__()

        self.module_list = nn.ModuleList()
        for _ in range(layer_num-1):
            self.module_list.append(SynthesisLayer(ch, ch, style_dim,
                                                   resolution,
                                                   use_noise=False,
                                                   activation='lrelu',
                                                   resample_filter=None))
        self.module_list.append(SynthesisLayer(ch, out_ch, style_dim,
                                               resolution,
                                               use_noise=False,
                                               activation='tanh',
                                               resample_filter=None))


    def forward(self, x: torch.Tensor, s: torch.Tensor):
        for m in self.module_list:
            x = m(x, s)
        return x
