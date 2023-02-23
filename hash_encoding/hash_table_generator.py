"""This is the hash-table generator network based on """
from typing import List

import numpy as np

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from utils.utils import sinuous_pos_encode
from utils.dist_utils import dprint
from hash_encoding.modules import StylelizedTransformerBlock
from hash_encoding.modules import HashUp
from hash_encoding.modules import HashSideOut
from hash_encoding.modules import StackedModulatedMLP
from hash_encoding.layers import ModulatedLinear


class HashTableGenerator(nn.Module):
    def __init__(self,
                 z_dim: int,
                 table_num: int,
                 table_size_log2: int,
                 init_dim: int,
                 res_min: int,
                 res_max: int,
                 style_dim: int=256,
                 head_dim: int=64,
                 use_layer_norm: bool=False,
                 learnable_side_up: bool=False,
                 fixed_random: bool=False,
                 linear_up: bool=True,
                 resample_filter=[1,3,3,1],
                 output_skip: bool=True,
                 shuffle_input: bool=False,
                 spatial_atten: bool=False,
                 ):
        """
            Args:
                z_dim (int): latent vector dimension
                table_num (int): how many tables (scales) we want to use.
                table_size_log2 (int): the final size will be 2 ** table_size_log2
                init_dim (int): initial size of the each token
                res_max (int): maximum resolution
                res_min (int): minimum resolution
                style_dim (int): mapped style code dimension. (default: 256)
                head_dim (int): head_dim, head dimension for a single unit.
                    (default: 64)
                use_layer_norm (bool): Whether to use layer normalization in 
                    transformer. (default: False)
                learnable_side_up (bool): The side upsample weight is learnbale
                    or not. (default: False)
                fixed_random (bool): The fixed weight is randomized or not.
                resample_filter (List): Low pass filter
                output_skip (bool): If use output skip. (default: True)
                shuffle_input (bool): shuffle input of each block according to 
                    indices (default: False)
                spatial_atten (bool): Spatial attention or not. (default: False)
        """
        super(HashTableGenerator, self).__init__()

        self.z_dim = z_dim
        self.table_num = table_num
        self.table_size_log2 = table_size_log2
        self.init_dim = init_dim
        self.res_max = res_max
        self.res_min = res_min
        self.head_dim = head_dim
        self.style_dim = style_dim
        self.use_layer_norm = use_layer_norm
        self.learnable_side_up = learnable_side_up
        self.fixed_random = fixed_random
        self.linear_up = linear_up
        self.output_skip = output_skip
        self.shuffle_input = shuffle_input
        self.spatial_atten= spatial_atten
        self.F = 2 #NOTE: We only support entry size 2 now for CUDA programming reason
        self.levels = int(self.table_size_log2 - np.log2(self.init_dim))
        self.data_dim = 2 # Is this a 2D data or 3D data

        self.b_res = np.exp((np.log(res_max) - np.log(res_min)) / (self.levels))

        # NOTE: we forcefully set 
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(table_num,
                                                self.init_dim))
        self._build_layers()
        dprint('Finished building hash table generator.', color='g')

    @property
    def s_num(self) -> int:
        return self.levels + 3

    def _build_layers(self) -> None:
        """This is where to put layer building."""
        # Upsample levels to go
        input_dim = self.init_dim

        for i in range(self.levels + 1):
            dim_now = int(input_dim * 2 ** i)
            head_dim_now = min(dim_now, self.head_dim)
            # The dimension for a single head remains constant
            nhead_now = max(dim_now // head_dim_now, 1)
            # Use the dim_now to compute block_num and sample_size
            block_num = 2
            # Sample size and sample res
            sample_size = min(dim_now // 2, 128)
            sample_res = int(self.res_min * self.b_res ** i)
            # Every transformer block has 2 transform layers
            transform_block = StylelizedTransformerBlock(dim_now,
                                                         head_dim_now,
                                                         self.table_num,
                                                         self.style_dim,
                                                         self.res_min,
                                                         self.res_max,
                                                         block_num=block_num,
                                                         hidden_dim=dim_now,
                                                         activation=nn.ReLU,
                                                         shuffle_input=self.shuffle_input,
                                                         use_prob_attention=False,
                                                         spatial_atten=self.spatial_atten,
                                                         sample_size=sample_size,
                                                         sample_res=sample_res,
                                                         only_linear=False)
            setattr(self, f'transformer_block_{i}', transform_block)

    def freeze_until_level(self, freeze_level)-> None:
        for i in range(freeze_level):
            getattr(self, f'transformer_block_{i}').requires_grad_(False)
    
    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
            Args:
                s1 (torch.Tensor): B x N_S x S_DIM (The W, the mapped style code)
                s2 (torch.Tensor): possible additional style code
            Return:
                hash_tables (torch.Tensor): B x H_N x H_S x (F=2)
                out_level (int): If not None, the level will return at out_level
                    and only train out_level + 1
        """
        b = s1.size(0)
        s1_iter = iter(s1.unbind(dim=1)) # The across table one
        s2_iter = iter(s2.unbind(dim=1)) # THe along table one
        x = self.pos_encoding.repeat(b, 1, 1)

        # Transformers following
        for i in range(self.levels+1):
            x = getattr(self, f'transformer_block_{i}')(x,
                                                        next(s1_iter),
                                                        next(s2_iter))
            # Upscale or output
            if i != self.levels:
                # Upscale as concat
                x = torch.cat((x, x), dim=-1)
            else:
                out = x

        hash_tables = out.reshape(out.shape[0], out.shape[1],
                                  out.shape[2] // self.F,
                                  self.F)

        return hash_tables



if __name__ == '__main__':
    h_net = HashTableGenerator(table_num=16,
                               table_size_log2=13,
                               init_dim=16,
                               res_max=512,
                               res_min=16,
                               style_dim=256,
                               head_dim=64).cuda()

    s = torch.randn(4, 256).cuda()
    out = h_net(s)
    print(out.shape)
