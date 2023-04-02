"""
Modules for generating hash tables
"""
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.ops import bias_act
from training.networks_stylegan2 import FullyConnectedLayer
from training.networks_stylegan2 import modulated_conv2d
from training.networks_stylegan2 import SynthesisLayer
from training.networks_stylegan2 import Conv2dLayer

from hash_encoding.layers import ModulatedLinear
from hash_encoding.layers import TokenWiseModulatedLinear
from hash_encoding.layers import ModulatedGridLinear
from hash_retrieve_module import HashTableRetrieve
from hash_encoding.prob_attention import ProbAttention
from hash_encoding.other_networks import MultiHeadOffsetNetwork
from utils.utils import get_shuffle_table_indices
from utils.utils import render
from utils.utils import sample_coords


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 head_dim: int,
                 s_dim: int,
                 token_num: int,
                 use_prob_attention: bool=False,
                 hidden_dim: int=None,
                 out_dim: int=None,
                 activation: nn.Module=nn.GELU,
                 deformable_head_num: int=8,
                 ):
        """
            Args:
                feat_dim (int): The token dimension of the input
                head_num (int): The head number
                s_dim (int): style mapping size
                use_prob_attention (bool): Whether to use probability attention
                    (default: False)
                hidden_dim (int): hidden dimension
                out_dim (int): out dimension
                activation (nn.Module): possible activation function
                    (default: nn.GELU)
                deformable_flag (bool): Whether to do deformable.
                    (default: False)
                deformable_head_num (int): 
        """
        super().__init__()
        self.token_num = token_num
        self.feat_dim = feat_dim
        self.s_dim = s_dim
        self.hidden_dim = feat_dim if hidden_dim is None else hidden_dim
        self.out_dim = feat_dim if out_dim is None else out_dim
        self.head_dim = head_dim
        self.head_num = self.hidden_dim // head_dim
        self.deformable_head_num = deformable_head_num

        self.k_mapping = nn.Linear(feat_dim, self.hidden_dim)
        self.q_mapping = nn.Linear(feat_dim, self.hidden_dim)
        self.v_mapping = nn.Linear(feat_dim, self.hidden_dim)
        self.o_mapping = nn.Linear(
                    self.hidden_dim * self.deformable_head_num, self.out_dim)

        # Deformable Part
        self.offset_network = MultiHeadOffsetNetwork(self.feat_dim,
                                                     self.feat_dim,
                                                     token_num,
                                                     deformable_head_num,
                                                     activation)

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
        q = self._resize_head_to_batch(q)
        k = self._resize_head_to_batch(k)
        v = self._resize_head_to_batch(v)
        # A is of shape B x N x N
        A = F.softmax(torch.einsum('bnc,bkc->bnk', q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bnk,bkc->bnc', A, v) 
        return out

    def _resample_tokens(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: B x N x C
                offsets: # B x (DH_NUM x T) x C x 2
        """
        b = x.size(0)
        c = x.size(2)

        x = x.unsqueeze(dim=1).repeat(1, 1, self.deformable_head_num, 1)
        x = F.grid_sample(x, offsets, mode='bilinear', align_corners=True) # B x 1 x (DH_NUM x T) x C
        x = x.reshape(int(b*self.deformable_head_num), self.token_num, c)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: B x T x C
        """
        b = x.size(0) # Original batch size

        offsets = self.offset_network(x) # B x (DH_NUM x T) x C x 2
        x = self._resample_tokens(x, offsets) # (B x DH_NUM) x T x C

        # (B x H) x N x C
        batch_size = x.size(0) # NOTE: The batch size here is augmented
                               #    as with the number of deformable head
        token_num = x.size(1)
        q = self.q_mapping(x).reshape(batch_size, token_num, self.head_num, self.head_dim)
        k = self.k_mapping(x).reshape(batch_size, token_num, self.head_num, self.head_dim)
        v = self.v_mapping(x).reshape(batch_size, token_num, self.head_num, self.head_dim)

        out = self._resize_head_back(self._scaled_dot_product_attention(q, k, v),
                                     batch_size,
                                     token_num)
        out = out.reshape(b, self.deformable_head_num, self.token_num, self.hidden_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, self.token_num, self.hidden_dim * self.deformable_head_num)
        out = self.o_mapping(out)

        return out # b x N x C

    def extra_repr(self) -> str:
        return (f'Input dimension {self.feat_dim}; Head number {self.head_num} '
                f'Style dimension {self.s_dim}')


class StackedModulatedMLP(nn.Module):
    def __init__(self, in_ch: int, h_ch: int, out_ch: int, s_dim: int,
                 n_layers: int,
                 table_num: int=16,
                 in_activ: nn.Module=nn.ReLU,
                 out_activ: nn.Module=nn.Tanh,
                 tokenwise_linear: bool=False,
                 use_layer_norm: bool=False):
        """
            Args:
                in_ch: input dimension
                h_ch: hidden dimension
                out_ch: output dimension
                s_dim: style code dimension
                n_layers: how many layers of MLPs in total
                    (including input and output layers)
                table_num (int): number of tables
                in_activ : inside (hidden layers) activation
                out_activ : output activation
                norm_layer (nn.Module): if Other normalization is used
                use_layer_norm (nn.Module): Use layer normalization
                tokenwise_linear (bool): If tokenwise linear or not.
                    (default: False)
        """
        super().__init__()

        self.module_list = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        linear_layer = partial(ModulatedLinear, table_num=table_num) if not tokenwise_linear else \
                        partial(TokenWiseModulatedLinear, table_num=table_num)

        for i in range(n_layers):
            if i == 0:
                self.module_list.append(linear_layer(in_ch, h_ch, s_dim, activation=in_activ))
            elif i == n_layers - 1:
                self.module_list.append(linear_layer(h_ch, out_ch, s_dim, activation=out_activ))
            else:
                self.module_list.append(linear_layer(h_ch, h_ch, s_dim, activation=in_activ))

    def forward(self, x, s):
        for m in self.module_list:
            x = m(x, s)
            if self.use_layer_norm:
                x = F.layer_norm(x, [x.size(-1),])
        return x


class StylelizedTransformerBlock(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 head_dim: int,
                 table_num: int,
                 s_dim: int,
                 res_min: int,
                 res_max: int,
                 block_num: int=1,
                 activation: nn.Module=nn.GELU,
                 sample_size: int=64,
                 sample_res: int=None,
                 use_layer_norm=True,
                 upsample=False,
                 use_prob_attention=False,
                 hidden_dim: int=None,
                 shuffle_input: bool=False,
                 spatial_atten: bool=False,
                 only_linear: bool=False,
                 tokenwise_linear: bool=False,
                 no_norm_layer: bool=False,
                 deformable_head_num: int=8,
                 ):
        """
            Args:
                feat_dim: The token dimension of the input
                head_dim: Head dimension
                table_num: The table number of the token
                s_dim: style mapping size
                res_min (int): minimum resolution
                res_max (int): maximum resolution
                block_num: How many multihead attention block will in one
                    Transformer block.
                activation: activation function. (default: nn.ReLU)
                sample_size (int): sample size to do spatial attention.
                sample_res (int): sample resolution if given. (default: None)
                use_layer_norm (bool): Whether to use layer normalization in 
                    transformer. (default: False)
                hidden_dim (int): Hidden dimension of attention. 
                shuffle_input (bool): Whether to shuffle input according to some indices
                spatial_atten (bool): Use spatial attention or not. 
                    (default: False)
                only_linear (bool): Only use linear operation in this block, 
                    no attention block is applied.
                tokenwise_linear (bool): If we use tokenwise linear or not.
                    (default: False)
                no_norm_layer (bool): No normalization layer.
                    (default: False)
                deformable_head_num (int): Deformable head number.
                    (default: 8)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.head_dim = head_dim
        self.table_num = table_num
        self.s_dim = s_dim
        self.block_num = block_num
        self.upsample = upsample
        self.res_min = res_min
        self.res_max = res_max
        self.sample_size = sample_size # Not used for now
        self.sample_res = sample_res
        self.shuffle_input = shuffle_input
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.use_prob_attention = use_prob_attention
        self.hidden_dim = hidden_dim
        self.spatial_atten = spatial_atten
        self.only_linear = only_linear
        self.tokenwise_linear = tokenwise_linear
        self.no_norm_layer = no_norm_layer
        self.deformblae_head_num = deformable_head_num

        if shuffle_input:
            random_indices = get_shuffle_table_indices(table_num, feat_dim)
            self.register_buffer('random_indices', random_indices)

        self._build_blocks()

    def _build_blocks(self):
        self.t_blocks = nn.ModuleList()
        self.l_layers = nn.ModuleList()
        for _ in range(self.block_num):
            # The attention or cross table-block
            if self.only_linear: t_block = nn.Identity();
            else:
                if not self.spatial_atten:
                    t_block = MultiHeadAttention(
                        self.feat_dim,
                        self.head_dim,
                        self.s_dim,
                        self.table_num,
                        hidden_dim=self.hidden_dim,
                        activation=self.activation,
                        deformable_head_num=self.deformblae_head_num)
                else:
                    t_block = HashAttention(self.table_num,
                                            self.res_min,
                                            self.sample_res,
                                            self.s_dim,
                                            self.sample_size,
                                            self.head_dim,
                                            sample_res=self.sample_res,
                                            activation=self.activation)
            self.t_blocks.append(t_block)

            # The linear or along table block
            self.l_layers.append(
                StackedModulatedMLP(self.feat_dim,
                                    self.feat_dim,
                                    self.feat_dim,
                                    self.s_dim,
                                    2,
                                    in_activ=self.activation,
                                    out_activ=nn.Identity,
                                    tokenwise_linear=self.tokenwise_linear,
                                    table_num=self.table_num)
            )
            

    def forward(self,
                x: torch.Tensor,
                s1: torch.Tensor,
                s2: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: input hash table. B x N x C
                s1: style code across the table
                s2: style code along the table
        """

        batch_size = x.size(0)
        if self.shuffle_input:
            x = torch.gather(x, -1, self.random_indices.repeat(batch_size, 1, 1))
        # Shuffle within hash tables
        for t, l in zip(self.t_blocks, self.l_layers):
            x = F.layer_norm(t(x) + x, (self.feat_dim,))
            x = F.layer_norm(l(x, s2) + x, (self.feat_dim,))
        return x
        
    
    def extra_repr(self) -> str:
        return (f'Input dimension {self.feat_dim}; Head number {self.head_num} '
                f'Style dimension {self.s_dim}; Block number {self.block_num}')


class StackedModulatedGridLinear(nn.Module):
    def __init__(self, in_ch: int, s_dim: int, token_num: int,
                 layer_n: int,
                 sample_res: int=256,
                 activation: nn.Module=nn.ReLU,
                 add_pos_encodings: bool=False,
                 inter_filter: bool=False,
                 next_token_num: int=None,
                 upsample: bool=False
                ):

        super().__init__()
        self.in_ch = in_ch

        self.mgl_list = nn.ModuleList()
        for i in range(layer_n):
            self.mgl_list.append(ModulatedGridLinear(
                    in_ch,
                    in_ch,
                    s_dim,
                    token_num,
                    activation=activation,
                    inter_filter=inter_filter,
                    sample_res=sample_res,
                    add_pos_encodings=True if (add_pos_encodings and i==0) else False,
                    next_token_num=next_token_num if i==layer_n-1 else None,
                    ))


    def forward(self, x, s):
        for l in self.mgl_list:
            #x = F.layer_norm(l(x, s) + x, (self.in_ch,))
            x = l(x, s)
        return x

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
        self.table_num = table_num

        #self.m_linear = StackedModulatedMLP(table_num*2, 32, 32, style_dim,
        #                                    n_layers=2,
        #                                    in_activ=nn.ReLU,
        #                                    out_activ=nn.ReLU)
        self.conv2d = SynthesisLayer(table_num*2, 3, style_dim, res_max,
                                     use_noise=False,
                                     activation='linear',
                                     conv_clamp=256)
        coords = sample_coords(None, res_max, single_batch=True)
        self.register_buffer('coords', coords)

    def forward(self, x, s) -> torch.Tensor:
        """
            Args:
                hash_tables: the hash tables B x H_N x H_S
                coords: sampled coordiates, B x N x (2 or 3)
                s: styled code B x S_DIM
        """
        b = x.shape[0]
        hash_tables = x.reshape(b, x.shape[1], x.shape[2] // 2, 2)
        hash_retrieved_feats = HashTableRetrieve.apply(hash_tables,
                                                       self.coords.repeat(b, 1, 1),
                                                       self.res_min,
                                                       self.res_max)
        #feats = self.m_linear(hash_retrieved_feats, s)
        if self.coords.shape[-1] == 2:
            conv_feats = hash_retrieved_feats.reshape(b,
                                                      self.res_max,
                                                      self.res_max,
                                                      self.table_num*2
                                                      ).permute(0, 3, 1, 2)
            return self.conv2d(conv_feats, s)
        elif self.coords.shape[-1] == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
