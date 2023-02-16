"""Layers to generate the hash table."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.networks_stylegan2 import FullyConnectedLayer

class ModulatedLinear(nn.Module):
    def __init__(self, in_ch, out_ch, s_dim,
                 activation=None,
                 bias=True):
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
        batch_size = x.shape[0]
        s = self.s_mapping(s)

        weight = self.weight
        w = weight.unsqueeze(dim=0) # 1 x OUT x IN
        w = w * s.reshape(batch_size, 1, -1)
        decoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt() # B x O

        x = x * s.reshape(batch_size, 1, -1)
        x = F.linear(x, weight, bias=self.bias) # B x (N) x O
        x = x * decoefs.reshape(batch_size,  1, -1)
        if self.activ is not None:
            x = self.activ(x)

        return x


class TokenWiseModulatedLinear(nn.Module):
    def __init__(self, in_ch, out_ch, table_num, s_dim,
                 activation=None,
                 bias=True):
        """
            Args:
                in_ch: input channel (has not been divided 2)
                out_ch: output channel
                table_num: table number (token number)
                s_dim: style dimension

            ..note.: This espacially for the computation of hash table.
        """
        super().__init__()

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
        table_num = x.shape[1]

        s = self.s_mapping(s)

        weight = self.weight  # table_num x O x I
        w = weight.unsqueeze(dim=0)
        w =  w * s.reshape(batch_size, -1, 1, 1)
        decoefs = (w.square().sum(dim=[3]) + 1e-8).rsqrt() # B x table_num x O

        x = x * s.reshape(batch_size, -1, 1)
        x = torch.einsum('noc,bnc->bno', weight, x).contiguous()
        x = x * decoefs

        if self.bias is not None:
            x = x + self.bias

        if self.activ is not None:
            x = self.activ(x)

        return x
