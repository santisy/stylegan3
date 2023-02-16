"""
Modules for generating hash tables
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.ops import bias_act
from training.networks_stylegan2 import FullyConnectedLayer
from training.networks_stylegan2 import modulated_conv2d

from hash_encoding.layers import ModulatedLinear
from hash_encoding.layers import TokenWiseModulatedLinear
from hash_retrieve_module import HashTableRetrieve
from hash_retrieve_module import HashTableRecon
from utils.utils import sample_coords
from utils.utils import render


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int, head_num: int, s_dim: int,
                 hidden_dim: int=None,
                 out_dim: int=None,
                 activation=nn.Identity):
        """
            Args:
                feat_dim: The token dimension of the input
                head_num: The head number
                s_dim: style mapping size
                hidden_dim: hidden dimension
                out_dim: out dimension
                activation: possible activation function
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.head_num = head_num
        self.s_dim = s_dim
        self.hidden_dim = feat_dim if hidden_dim is None else hidden_dim
        self.out_dim = feat_dim if out_dim is None else out_dim
        self.head_dim = self.hidden_dim // head_num

        self.k_mapping = ModulatedLinear(feat_dim, hidden_dim, s_dim)
        self.q_mapping = ModulatedLinear(feat_dim, hidden_dim, s_dim)
        self.v_mapping = ModulatedLinear(feat_dim, hidden_dim, s_dim,
                                         activation=activation)
        self.o_mapping = ModulatedLinear(hidden_dim, out_dim, s_dim)

    def _resize_head_to_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x (torch.Tensor): B x N x H x C, H is the head number
        """ 
        batch_size = x.shape[0]
        token_size = x.shape[1]
        head_num = x.shape[2]
        channel_size = x.shape[3]
        return x.permute(0, 2, 1, 3).reshape(batch_size * head_num,
                                             token_size,
                                             channel_size)

    def _resize_head_back(self, x: torch.Tensor,
                          batch_size: int,
                          token_num: int) -> torch.Tensor:
        """
            Args:
                x (torch.Tensor): (B x H) x N x C
                batch_szie (int): batch size
                token_num (int): token number
        """
        x = x.reshape(batch_size, self.head_num, token_num, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, token_num, -1)
        return x

    def _scaled_dot_product_attention(self,
                                      q: torch.Tensor,
                                      k: torch.Tensor,
                                      v: torch.Tensor) -> torch.Tensor:
        """
            Args:
                q, k, v are all torch.Tensor and shapes of B x N x C
        """
        # A is of shape B x N x N
        A = F.softmax(torch.einsum('bnc,bkc->bnk', q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bnk,bkc->bnc', A, v) 
        return out

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: B x N x FEAT_DIM
                s: B x S_DIM
        """
        batch_size = x.shape[0]
        token_num = x.shape[1]

        # (B x H) x N x C
        k = self._resize_head_to_batch(self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim))
        q = self._resize_head_to_batch(self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim))
        v = self._resize_head_to_batch(self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim))

        out = self._resize_head_back(self._scaled_dot_product_attention(k, q, v),
                                     batch_size,
                                     token_num)
        
        out = self.o_mapping(out, s)

        return out

    def extra_repr(self) -> str:
        return (f'Input dimension {self.feat_dim}; Head number {self.head_num} '
                f'Style dimension {self.s_dim}')


class HashAttention(nn.Module):
    def __init__(self, table_num: int, res_min: int, res_max: int,
                 style_dim: int,
                 sample_size: int,
                 head_num: int,
                 resample_filter=[1,3,3,1]):
        """
            Hash Attention: 
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
        coords = sample_coords(None, res_max, sample_size=sample_size,
                               single_batch=True)
        self.register_buffer('coords', coords)

        # Attention k, q, v
        self.multi_head_attention = MultiHeadAttention(ch,
                                                       head_num,
                                                       style_dim,
                                                       hidden_dim=ch * head_num,
                                                       out_dim=ch,
                                                       activation=nn.LeakyReLU)

    def forward(self, inputs, s):

        batch_size = inputs.shape[0]
        device = inputs.device
        # The `2` here is the entry length in hash tables
        table_dim = inputs.shape[2] // 2
        table_num = inputs.shape[1]
        hash_tables = inputs.reshape(batch_size, table_num, table_dim, 2)
        coords = self.coords.repeat(batch_size, 1, 1)

        # Hash out to image tensor
        hash_retrieved_feats = HashTableRetrieve.apply(hash_tables,
                                                       coords,
                                                       self.res_min,
                                                       self.res_max)

        # Self attention
        tokenize_tensor = hash_retrieved_feats
        tokenize_tensor = F.layer_norm(
            self.multi_head_attention(tokenize_tensor, s) + tokenize_tensor,
            [self.ch])
        
        # Recon the hash tables
        recon_hash_tables = HashTableRecon.apply(tokenize_tensor,
                                                 coords,
                                                 table_dim,
                                                 self.res_min,
                                                 self.res_max)

        # FIXME: The normalizing should be inside Hash Table Recon
        return recon_hash_tables.reshape(batch_size, table_num, -1)


class StackedModulatedMLP(nn.Module):
    def __init__(self, in_ch: int, h_ch: int, out_ch: int, s_dim: int,
                 n_layers: int,
                 in_activ=nn.ReLU,
                 out_activ=nn.Tanh,
                 norm_layer=nn.Identity):
        """
            Args:
                in_ch: input dimension
                h_ch: hidden dimension
                out_ch: output dimension
                s_dim: style code dimension
                n_layers: how many layers of MLPs in total
                    (including input and output layers)
                in_activ : inside (hidden layers) activation
                out_activ : output activation
                norm_layer (nn.Module): if Other normalization is used
        """
        super().__init__()

        self.module_list = nn.ModuleList()
        self.norm_layer = norm_layer

        for i in range(n_layers):
            if i == 0:
                self.module_list.append(ModulatedLinear(in_ch, h_ch, s_dim, activation=in_activ))
            elif i == n_layers - 1:
                self.module_list.append(ModulatedLinear(h_ch, out_ch, s_dim, activation=out_activ))
            else:
                self.module_list.append(ModulatedLinear(h_ch, h_ch, s_dim, activation=in_activ))

    def forward(self, x, s):
        for m in self.module_list:
            x = m(x, s)
        return x


class StylelizedTransformerBlock(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 head_num: int,
                 table_num: int,
                 s_dim: int,
                 res_min: int,
                 res_max: int,
                 block_num: int=1,
                 activation=nn.ReLU,
                 sample_size:int=64,
                 use_layer_norm=True,
                 upsample=False,
                 no_pointwise_linear=False):
        """
            Args:
                feat_dim: The token dimension of the input
                head_num: The head number
                table_num: The table number of the token
                s_dim: style mapping size
                res_min (int): minimum resolution
                res_max (int): maximum resolution
                block_num: How many multihead attention block will in one
                    Transformer block.
                activation: activation function. (default: nn.ReLU)
                sample_size (int): sample size to do spatial attention.
                use_layer_norm (bool): Whether to use layer normalization in 
                    transformer. (default: False)
                no_pointwise_linear (bool): Do not use any pointwise linear
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.head_num = head_num
        self.table_num = table_num
        self.s_dim = s_dim
        self.block_num = block_num
        self.upsample = upsample
        self.res_min = res_min
        self.res_max = res_max
        self.sample_size = sample_size
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.no_pointwise_linear = no_pointwise_linear

        self._build_blocks()

    def _build_blocks(self):
        self.t_blocks = nn.ModuleList()
        self.l_layers = nn.ModuleList()
        for _ in range(self.block_num):
            self.t_blocks.append(HashAttention(
                self.table_num,
                self.res_min,
                self.res_max,
                self.s_dim,
                sample_size=self.sample_size,
                head_num=self.head_num
            ))
            self.l_layers.append(TokenWiseModulatedLinear(
                self.feat_dim,
                self.feat_dim,
                self.table_num,
                self.s_dim,
                activation=self.activation
            ) if not self.no_pointwise_linear else nn.Identity())
            

    def forward(self, x, s):
        for t, l in zip(self.t_blocks, self.l_layers):
            x = F.layer_norm(t(x, s) + x, (self.feat_dim,))
            if not self.no_pointwise_linear:
                x = F.layer_norm(l(x, s) + x, (self.feat_dim,))
        return x
        
    
    def extra_repr(self) -> str:
        return (f'Input dimension {self.feat_dim}; Head number {self.head_num} '
                f'Style dimension {self.s_dim}; Block number {self.block_num}')



class HashUp(nn.Module):
    def __init__(self,
                 table_num: int,
                 input_dim: int,
                 res_min: int=None,
                 res_max: int=None,
                 learnable: bool=True,
                 fixed_random: bool=False
                 ):
        """
            Args:
                table_num (int): table number (token number)
                input_dim (int): input dimension of token
                res_min (int): minimum resolution
                res_max (int): maximum resolution
                learnable (bool): If the upsample mapping is learnable or not
                    (default: False)
                fixed_random (bool): the weight is randomized but fixed.
                    (default: False)
                res_list (List[float]): list of resolutions for each hash table
        """
        super().__init__()
        self.learnable = learnable
        weight = torch.zeros(table_num, input_dim * 2, input_dim)
        if learnable:
            self.register_parameter('weight',
                                    nn.Parameter(nn.init.xavier_normal_(weight)))
        else:
            if fixed_random:
                self.register_buffer('weight',
                                     nn.init.xavier_uniform_(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: token, size of B x N x C
        """
        if self.learnable:
            weight = self.weight
            decoefs = weight.square().sum(dim=[2], keepdim=True).rsqrt()
            w = weight * decoefs
            x = torch.einsum('bnc,noc->bno', x, w).contiguous()
        else:
            x = torch.cat((x, x), dim=2)
        return x


class HashSideOut(nn.Module):
    def __init__(self,
                 res_min: int,
                 res_max: int,
                 table_num: int,
                 style_dim: int):
        super().__init__()
        self.res_min = res_min
        self.res_max = res_max

        self.m_linear = ModulatedLinear(table_num*2, 3, style_dim)

    def forward(self, x, coords, s) -> torch.Tensor:
        """
            Args:
                hash_tables: the hash tables B x H_N x H_S
                coords: sampled coordiates, B x N x (2 or 3)
                s: styled code B x S_DIM
        """
        b = x.shape[0]
        hash_tables = x.reshape(b, x.shape[1], x.shape[2] // 2, 2)
        hash_retrieved_feats = HashTableRetrieve.apply(hash_tables,
                                                       coords,
                                                       self.res_min,
                                                       self.res_max)
        feats = self.m_linear(hash_retrieved_feats, s)
        if coords.shape[-1] == 2:
            return feats.reshape(b, self.res_max, self.res_max, 3).permute(0, 3, 1, 2)
        elif coords.shape[-1] == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
