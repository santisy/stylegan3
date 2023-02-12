"""
Modules for generating hash tables
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_encoding.layers import ModulatedLinear
from hash_encoding.layers import TokenWiseModulatedLinear
from hash_retrieve_module import HashTableRetrieve
from hash_retrieve_module import get_hash_up_mat


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int, head_num: int, table_num: int, s_dim: int):
        """
            Args:
                feat_dim: The token dimension of the input
                head_num: The head number
                table_num: The number of table (token number)
                s_dim: style mapping size
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.head_num = head_num
        self.head_dim = feat_dim // head_num
        self.s_dim = s_dim

        self.k_mapping = ModulatedLinear(feat_dim, feat_dim, s_dim)
        self.q_mapping = ModulatedLinear(feat_dim, feat_dim, s_dim)
        self.v_mapping = ModulatedLinear(feat_dim, feat_dim, s_dim)
        self.o_mapping = ModulatedLinear(feat_dim, feat_dim, s_dim)

    def forward(self, x, s):
        """
            Args:
                x: B x N x FEAT_DIM
                s: B x S_DIM
        """
        batch_size = x.shape[0]
        token_num = x.shape[1]

        # B x N x H x D
        k = self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim)
        q = self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim)
        v = self.k_mapping(x, s).reshape(batch_size, token_num, self.head_num, self.head_dim)

        A = F.softmax(torch.einsum('bnhd,bnjd->bnhj', q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bnhj,bnjd->bnhd', A, v).reshape(batch_size, token_num, self.feat_dim)
        out = self.o_mapping(out, s)

        return out

    def extra_repr(self) -> str:
        return (f'Input dimension {self.feat_dim}; Head number {self.head_num} '
                f'Style dimension {self.s_dim}')


class StackedModulatedMLP(nn.Module):
    def __init__(self, in_ch: int, h_ch: int, out_ch: int, s_dim: int,
                 n_layers: int,
                 in_activ=nn.LeakyReLU,
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
    def __init__(self, feat_dim: int, head_num: int, table_num: int, s_dim: int,
                 block_num: int=1,
                 activation=nn.ReLU,
                 use_layer_norm=True,
                 upsample=False):
        """
            Args:
                feat_dim: The token dimension of the input
                head_num: The head number
                table_num: The table number of the token
                s_dim: style mapping size
                block_num: How many multihead attention block will in one
                    Transformer block.
                activation: activation function. (default: nn.ReLU)
                use_layer_norm (bool): Whether to use layer normalization in 
                    transformer. (default: False)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.head_num = head_num
        self.table_num = table_num
        self.s_dim = s_dim
        self.block_num = block_num
        self.upsample = upsample
        self.activation = activation
        self.use_layer_norm = use_layer_norm

        self._build_blocks()

    def _build_blocks(self):
        self.t_blocks = nn.ModuleList()
        self.l_layers = nn.ModuleList()
        for _ in range(self.block_num):
            self.t_blocks.append(MultiHeadAttention(self.feat_dim,
                                                    self.head_num,
                                                    self.table_num,
                                                    self.s_dim))
            self.l_layers.append(ModulatedLinear(self.feat_dim,
                                                 self.feat_dim,
                                                 self.s_dim,
                                                 activation=self.activation,
                                                 bias=True
            ))
            

    def forward(self, x, s):
        for t, l in zip(self.t_blocks, self.l_layers):
            x = F.layer_norm(t(x, s) + x, (self.feat_dim,))
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
            else:
                assert res_min is not None and res_max is not None
                self.register_buffer('weight',
                                     get_hash_up_mat.get_hash_interp_matrix_2D(
                                        input_dim,
                                        table_num,
                                        res_min,
                                        res_max,
                                        2
                                     ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: token, size of B x N x C
        """
        weight = self.weight
        if self.learnable:
            decoefs = weight.square().sum(dim=[2], keepdim=True).rsqrt()
            w = weight * decoefs
        else:
            w = weight
        x = torch.einsum('bnc,noc->bno', x, w).contiguous()
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
