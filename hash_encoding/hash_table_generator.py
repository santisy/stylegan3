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
from hash_encoding.layers import ModulatedLinear

SAMPLE_SIZE = 128

class HashTableGenerator(nn.Module):
    def __init__(self,
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
                 ):
        """
            Args:
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
        """
        super(HashTableGenerator, self).__init__()

        self.table_num = table_num
        self.table_size_log2 = table_size_log2
        self.res_max = res_max
        self.res_min = res_min
        self.head_dim = head_dim
        self.style_dim = style_dim
        self.use_layer_norm = use_layer_norm
        self.learnable_side_up = learnable_side_up
        self.fixed_random = fixed_random
        self.linear_up = linear_up
        self.output_skip = output_skip
        self.F = 2 #NOTE: We only support entry size 2 now for CUDA programming reason

        b = np.exp((np.log(res_max) - np.log(res_min)) / (table_num - 1))
        res_list = res_min * b ** torch.arange(table_num)
        res_list = res_list.unsqueeze(dim=0) # 1 x H_N
        dprint(f'Hash table resolution list is {res_list}')

        init_dim = table_num
        self.init_dim = table_num
        self.levels = L = int(self.table_size_log2 - np.log2(init_dim))
        self.init_token_num = init_token_num = int(table_num * 2 ** L)
        # NOTE: we forcefully set 
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(init_token_num,
                                                self.init_dim))
        self._build_layers()

    def _get_block_num(self, dim_now: int) -> List[int]:
        table_dim = dim_now // 2
        return int(max(np.log2(table_dim / SAMPLE_SIZE), 1))

    def _build_layers(self) -> None:
        """This is where to put layer building."""
        # Upsample levels to go
        input_dim = self.init_dim
        init_token_num = self.init_token_num
        L = self.levels

        for i in range(L+1):
            dim_now = int(input_dim * 2 ** i)
            head_dim_now = min(dim_now, self.head_dim)
            # The dimension for a single head remains constant
            nhead_now = max(dim_now // head_dim_now, 1)
            # Use the dim_now to compute block_num and sample_size
            block_num = 2
            # Use ProbAttention or not 
            token_num_now = int(init_token_num / (2 ** i))
            # Every transformer block has 2 transform layers
            transform_block = StylelizedTransformerBlock(dim_now,
                                                         nhead_now,
                                                         self.table_num,
                                                         self.style_dim,
                                                         self.res_min,
                                                         self.res_max,
                                                         block_num=block_num,
                                                         activation=nn.ReLU,
                                                         use_prob_attention=True)
            setattr(self, f'transformer_block_{i}', transform_block)

    def _sample_coords(self, b, res_now):
        # 2D sampling case
        c = torch.arange(res_now) + 0.5
        x, y = torch.meshgrid(c, c)
        coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2)
        coords = coords.repeat(b, 1, 1) / res_now # Normalize it to [0, 1]

        return coords

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
            Args:
                s (torch.Tensor): B x S_DIM (The W, the mapped style code)

            Return:
                hash_tables (torch.Tensor): B x H_N x H_S x (F=2)
        """
        b = s.shape[0]
        s_iter = iter(s.unbind(dim=1))

        x = self.pos_encoding.repeat(b, 1, 1)
        # Transformers following
        for i in range(self.levels+1):
            x = getattr(self, f'transformer_block_{i}')(x, next(s_iter))
            if i != self.levels:
                n = x.size(1)
                x = x.reshape(b, n//2, -1)
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
