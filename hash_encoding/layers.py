"""Layers to generate the hash table."""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.networks_stylegan2 import SynthesisLayer
from training.networks_stylegan2 import FullyConnectedLayer
from hash_encoding.fourier_1d import SpectralConv1d
from hash_retrieve_module import HashTableRetrieve
from hash_retrieve_module import HashTableRecon
from utils.utils import sinuous_pos_encode
from utils.utils import sample_coords
from utils.utils import get_hash_mask

class AdaIN(nn.Module):
    def __init__(self, ch, s_dim):
        super().__init__()
        self.linear_map = nn.Linear(s_dim, 2 * ch, bias=False)

    def forward(self, x, s):
        a, b = self.linear_map(s).chunk(2, dim=-1)
        a = a.unsqueeze(dim=1) + 1.0
        b = b.unsqueeze(dim=1)
        x = F.layer_norm(x, (x.size(-1),))
        x = x * a + b


        return x

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


class HashFilter(nn.Module):
    def __init__(self, table_num: int, res_min: int, res_max: int,
                 style_dim: int,
                 sample_size: int,
                 head_dim: int,
                 sample_res: int=None,
                 activation: nn.Module=nn.GELU,
                 resample_filter=[1,3,3,1]):
        """
            Hash Filtering using conv/filter: 
                We first hash to spatial tensor chunk and do attention to this 
                    chunk and hash back to hash tables.
            Args:
                table_num (int): The number of tables (To determine the channel
                    dimension for convolution layer and style MLP)
                res_max (int): maximum resolution of hash tables
                res_min (int): minimum resolution of hash tables
                style_dim (int): style vector dimension
                sample_size (int): how many sampled points (locations)
                    under res max
                head_num (int): head number
                activation (nn.Module):  default nn.ReLU
                sample_res (int): sample resolution if given. (default: None)
                resample_filter (int): The low pass filter. 
                    (default: [1, 3, 3, 1])
        """
        super().__init__()
        self.res_min = res_min
        self.res_max = res_max
        self.sample_size = sample_size

        # The `2` here is the entry length in hash tables
        self.ch = ch = table_num * 2

        # Randomly sample coords here.
        self.sample_res = sample_res = res_max if sample_res is None else sample_res
        coords = sample_coords(None, sample_res, single_batch=True)
        self.register_buffer('coords', coords)

        self.conv1 = SynthesisLayer(self.ch,
                                    self.ch,
                                    style_dim,
                                    sample_res,
                                    kernel_size=3,
                                    up=1,
                                    use_noise=False,
                                    resample_filter=[1,3,3,1],
                                    activation='lrelu',
                                    )

    def forward(self, inputs, s):

        batch_size = inputs.size(0)
        # The `2` here is the entry length in hash tables
        table_dim = inputs.size(2) // 2
        table_num = inputs.size(1)
        hash_tables = inputs.reshape(batch_size, table_num, table_dim, 2)
        coords = self.coords.repeat(batch_size, 1, 1)

        # Hash out to image tensor
        hash_retrieved_feats = HashTableRetrieve.apply(hash_tables,
                                                       coords,
                                                       self.res_min,
                                                       self.res_max)

        block_tensor = hash_retrieved_feats.reshape(batch_size,
                                                    self.sample_res,
                                                    self.sample_res,
                                                    self.ch
                                                    ).permute(0, 3, 1, 2)

        block_tensor = self.conv1(block_tensor, s) 
        tokenize_tensor = block_tensor.reshape(batch_size, self.ch, -1
                                               ).permute(0, 2, 1).contiguous()
        
        # Recon the hash tables
        recon_hash_tables = HashTableRecon.apply(tokenize_tensor,
                                                 coords,
                                                 table_dim,
                                                 self.res_min,
                                                 self.res_max)

        outputs = recon_hash_tables.reshape(batch_size, table_num, -1) + inputs
        outputs = F.layer_norm(outputs, (inputs.size(-1),))
        return outputs


class AlongTokenLinear(nn.Module):
    def __init__(self,
                 ch: int,
                 s_dim: int,
                 activation: nn.Module=nn.ReLU,
                 bias: bool=True,
                 upsample: bool=False,
                 modes:  int=16,
                 ):
        """ This layer is a mimic of the inverse residual block of 
                mobile-netv2.
        """
        super().__init__()
        self.ch = ch
        self.modulated_spectral_conv = SpectralConv1d(ch, ch, s_dim, modes1=modes)

    def forward(self, x, s):
        x = self.modulated_spectral_conv(x, s)
        return x


class CrossTokenLinear(nn.Module):
    def __init__(self, ch: int, s_dim: int,
                 activation: nn.Module=nn.ReLU, bias: bool=True,
                 next_token_num: int=None,
                 no_modulated: bool=False):
        super().__init__()
        # Cross Token Linear
        self.ch = ch
        # We are shrinking the token number
        self.next_ch = next_ch = next_token_num if next_token_num is not None else ch
        self.linear1 = nn.Linear(ch, next_ch, bias=bias)
        self.linear2 = nn.Linear(next_ch, next_ch, bias=bias)
        self.activ1 = activation()
        self.activ2 = activation()



    def forward(self, x):
        ori_x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.activ1(x)
        x = self.linear2(x)
        x = self.activ2(x)
        return x


class ModulatedGridLinear(nn.Module):
    def __init__(self,
                 in_ch: int, out_ch: int, s_dim: int, token_num: int,
                 activation: nn.Module=nn.ReLU,
                 add_pos_encodings: bool=False,
                 sample_res: int=256,
                 res_min: int=16,
                 inter_filter: bool=False,
                 bias: bool=True,
                 upsample: bool = False,
                 next_token_num: int=None,
                 ):
        super().__init__()
        self.add_positional_encodings = add_pos_encodings
        self.inter_filter = inter_filter
        # Decrease token number and increase token length
        self.upsample = upsample 
        next_token_num = next_token_num if next_token_num is not None else token_num

        # Along Token linear
        self.along_linear = AlongTokenLinear(token_num, s_dim,
                                             activation,
                                             bias=bias,
                                             upsample=upsample)
        # TODO: write the masking here
        # hash_mask1 = get_hash_mask(sample_res, res_min, token_num,
        #                            in_ch * 2 if upsample else in_ch)
        # self.register_buffer('hash_mask1', hash_mask1)
        self.cross_linear = CrossTokenLinear(token_num, s_dim, activation,
                                             bias=bias,
                                             next_token_num=next_token_num,
                                             no_modulated=inter_filter)
        # hash_mask2 = get_hash_mask(sample_res, res_min, next_token_num,
        #                            in_ch * 2 if upsample else in_ch)
        # self.register_buffer('hash_mask2', hash_mask2)

        # Add positional encodings or not
        if add_pos_encodings:
            self.register_buffer('pos_encoding',
                                  sinuous_pos_encode(token_num, in_ch))
        
        # Using hash up and hash back at certain resolution and inter-mediate
        #   filter/conv
        if inter_filter:
            self.hash_filter = HashFilter(next_token_num, res_min, sample_res,
                                          s_dim, sample_res, None, None,
                                          activation=nn.ReLU)

        if next_token_num != token_num:
            self.short_cut = nn.Linear(token_num, next_token_num)
        else:
            self.short_cut = nn.Identity()

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: B x T x C
                s: B x S
        """
        batch_size = x.size(0)
        x_ori = x
        if self.add_positional_encodings:
            x = x + self.pos_encoding.repeat(batch_size, 1, 1)
        
        # Along token linear
        x = self.along_linear(x, s)
        x = self.cross_linear(x)
        if self.inter_filter:
            x = self.hash_filter(x, s)
        x_ori = self.short_cut(x_ori.permute(0, 2, 1))
        x = x + x_ori
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, (x.size(-1),))
        return x

