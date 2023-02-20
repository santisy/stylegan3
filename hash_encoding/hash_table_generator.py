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
                 shuffle_input: bool=False,
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
                shuffle_input (bool): shuffle input of each block according to 
                    indices (default: False)
        """
        super(HashTableGenerator, self).__init__()

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
        self.F = 2 #NOTE: We only support entry size 2 now for CUDA programming reason
        self.levels = int(self.table_size_log2 - np.log2(self.init_dim))

        b = np.exp((np.log(res_max) - np.log(res_min)) / (table_num - 1))
        res_list = res_min * b ** torch.arange(table_num)
        res_list = res_list.unsqueeze(dim=0) # 1 x H_N
        dprint(f'Hash table resolution list is {res_list}')

        # NOTE: we forcefully set 
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(table_num,
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

        for i in range(self.levels + 1):
            dim_now = int(input_dim * 2 ** i)
            head_dim_now = min(dim_now, self.head_dim)
            # The dimension for a single head remains constant
            nhead_now = max(dim_now // head_dim_now, 1)
            # Use the dim_now to compute block_num and sample_size
            block_num = 6
            # Use ProbAttention or not 
            token_num_now = int(init_token_num / (2 ** i))
            # Every transformer block has 2 transform layers
            transform_block = StylelizedTransformerBlock(dim_now,
                                                         nhead_now,
                                                         token_num_now,
                                                         self.style_dim,
                                                         self.res_min,
                                                         self.res_max,
                                                         block_num=block_num,
                                                         hidden_dim=dim_now*4,
                                                         activation=nn.ReLU,
                                                         shuffle_input=self.shuffle_input,
                                                         use_prob_attention=False)
            setattr(self, f'transformer_block_{i}', transform_block)

    def freeze_until_level(self, freeze_level)-> None:
        for i in range(freeze_level):
            getattr(self, f'transformer_block_{i}').requires_grad_(False)
    
    def forward(self, s: torch.Tensor, out_level: int=None) -> torch.Tensor:
        """
            Args:
                s (torch.Tensor): B x S_DIM (The W, the mapped style code)

            Return:
                hash_tables (torch.Tensor): B x H_N x H_S x (F=2)
                out_level (int): If not None, the level will return at out_level
                    and only train out_level + 1
        """
        if out_level is not None:
            self.freeze_until_level(out_level - 1)
        b = s.shape[0]
        s_iter = iter(s.unbind(dim=1))

        x = self.pos_encoding.repeat(b, 1, 1)
        pre_out = None # Previous level output

        # Transformers following
        for i in range(self.levels+1):
            # TOFIX: Is this detach necessary?
            if out_level is not None and i == out_level - 1:
                x = x.detach()
            pre_out = x

            x = getattr(self, f'transformer_block_{i}')(x, next(s_iter))

            if out_level is not None and i == out_level:
                out = x
                if i == 0:
                    pre_out = out
                break

            # Upscale or output
            if i != self.levels:
                out = torch.cat((x, x), dim=-1)
            else:
                out = x

        hash_tables = out.reshape(out.shape[0], out.shape[1],
                                  out.shape[2] // self.F,
                                  self.F)
        pre_hash_tables = out.reshape(pre_out.shape[0], pre_out.shape[1],
                                      pre_out.shape[2] // self.F,
                                      self.F)

        return hash_tables, pre_hash_tables



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
