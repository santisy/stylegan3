"""Layers to generate the hash table."""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.networks_stylegan2 import FullyConnectedLayer
from utils.utils import sinuous_pos_encode


class ModulatedLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s_dim: int,
                 activation: nn.Module=None,
                 bias: bool=True, **kwargs):
        super().__init__()
        weight = nn.Parameter(torch.randn(out_ch, in_ch))
        self.register_parameter('weight', weight)
        nn.init.xavier_normal_(self.weight)

        if bias:
            bias = nn.Parameter(torch.zeros(out_ch))
            self.register_parameter('bias', bias)
        else:
            self.bias = None
        
        if activation is not None:
            self.activ = activation()
        else:
            self.activ = None

        self.s_mapping = FullyConnectedLayer(s_dim, in_ch, bias_init=1)

    def forward(self, x, s):
        """
            x: B x (N) x IN
            s: B x s_dim

        """
        batch_size = x.size(0)
        s = self.s_mapping(s)
        # NOTE: The batch size may be different
        s_batch_size = s.size(0)
        if s_batch_size < batch_size:
            s = s.repeat(batch_size // s_batch_size, 1)

        weight = self.weight
        w = weight.unsqueeze(dim=0) # 1 x OUT x IN
        w = w * s.reshape(batch_size, 1, -1)
        decoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt() # B x O

        s = s.unsqueeze(dim=1) if x.dim() == 3 else s
        decoefs = decoefs.unsqueeze(dim=1) if x.dim() == 3 else decoefs

        x = x * s
        x = F.linear(x, weight, bias=self.bias) # B x (N) x O
        x = x * decoefs
        if self.activ is not None:
            x = self.activ(x)

        return x


class TokenWiseModulatedLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s_dim: int,
                 table_num: int=16,
                 activation: nn.Module=None,
                 bias: bool=True,
                 linear_clamp: float=256):
        """
            Args:
                in_ch: input channel (has not been divided 2)
                out_ch: output channel
                s_dim: style dimension
                table_num (int): table number (token number). (default: 16)
                activation (nn.Module): activation function.
                    (default: None)
                bias (bool): Use bias or not. (default: True)
                linear_clamp (float): value X, clamp the value of linear output
                    to [-X, +X]. (default: 256)

            ..note.: This espacially for the computation of hash table.
        """
        super().__init__()
        self.linear_clamp = linear_clamp
        self.lr_multiplier = 1.0 / np.sqrt(in_ch)

        weight = nn.Parameter(torch.randn(table_num, out_ch, in_ch))
        self.register_parameter('weight', weight)
        nn.init.xavier_normal_(self.weight)

        if bias:
            bias = nn.Parameter(torch.zeros(1, table_num, out_ch))
            self.register_parameter('bias', bias)
        else:
            self.bias = None

        if activation is not None:
            self.activ = activation()
        else:
            self.activ = None

        self.s_mapping = FullyConnectedLayer(s_dim, table_num, bias_init=1)

    def forward(self, x, s):
        batch_size = x.shape[0]

        s = self.s_mapping(s)

        weight = self.weight  # table_num x O x I
        w = weight.unsqueeze(dim=0) # B x table_num x O x I
        w =  w * s.reshape(batch_size, -1, 1, 1)
        decoefs = (w.square().sum(dim=[3,]) + 1e-8).rsqrt() # B x table_num x O

        x = x * s.reshape(batch_size, -1, 1)
        x = torch.einsum('noc,bnc->bno', weight, x).contiguous()
        x = x * decoefs

        if self.bias is not None:
            x = x + self.bias

        if self.activ is not None:
            x = self.activ(x)

        return x


class AlongTokenLinear(nn.Module):
    def __init__(self, ch: int, activation: nn.Module=nn.ReLU, bias: bool=True):
        super().__init__()
        self.ch = ch
        self.linear1 = nn.Linear(ch, ch, bias=bias)
        self.activ = activation()
        self.linear2 = nn.Linear(ch, ch, bias=bias)

    def forward(self, x):
        ori_x = x
        x = self.linear1(x)
        x = self.activ(x)
        x = self.linear2(x)
        x = F.layer_norm(x + ori_x, (self.ch,))
        return x


class CrossTokenLinear(nn.Module):
    def __init__(self, ch: int, s_dim: int,
                 activation: nn.Module=nn.ReLU, bias: bool=True):
        super().__init__()
        # Cross Token Linear
        self.ch = ch
        self.linear1 = ModulatedLinear(ch, ch, s_dim, activation=activation,
                                       bias=bias)
        self.linear2 = ModulatedLinear(ch, ch, s_dim, activation=None,
                                       bias=bias)

    def forward(self, x, s):
        ori_x = x
        x = x.permute(0, 2, 1)
        x = self.linear1(x, s)
        x = self.linear2(x, s)
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x + ori_x, (x.size(-1),))
        return x


class ModulatedGridLinear(nn.Module):
    def __init__(self,
                 in_ch: int, out_ch: int, s_dim: int, token_num: int,
                 activation: nn.Module=nn.ReLU,
                 add_pos_encodings: bool=False,
                 bias: bool=True
                 ):
        super().__init__()
        self.add_positional_encodings = add_pos_encodings

        # Along Token linear
        self.along_linear = AlongTokenLinear(in_ch, activation, bias=bias)
        self.cross_linear = CrossTokenLinear(token_num, s_dim, activation, bias=bias)

        # Add positional encodings or not
        if add_pos_encodings:
            self.register_buffer('pos_encoding',
                                  sinuous_pos_encode(token_num, in_ch))



    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: B x T x C
                s: B x S
        """
        batch_size = x.size(0)

        if self.add_positional_encodings:
            x = x + self.pos_encoding.repeat(batch_size, 1, 1)

        # Along token linear
        x = self.along_linear(x)
        x = self.cross_linear(x, s)

        return x
