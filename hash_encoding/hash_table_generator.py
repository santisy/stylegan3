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
from hash_encoding.modules import StackedModulatedGridLinear
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
                 tokenwise_linear: bool=False,
                 no_norm_layer: bool=False,
                 shrink_down: bool=False,
                 inter_filter: bool=False,
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
                tokenwise_linear (bool): If we use tokenwise linear or not.
                    (default: False)
                no_norm_layer (bool): No normalization layer.
                    (default: False)
                shrink_down (bool): If shrink down arch (default=False)
                inter_filter (bool): Internal filter using conv/filter.
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
        self.tokenwise_linear = tokenwise_linear
        self.no_norm_layer = no_norm_layer
        self.shrink_down = shrink_down
        self.inter_filter = inter_filter
        self.F = 2 #NOTE: We only support entry size 2 now for CUDA programming reason
        self.levels = int(self.table_size_log2 - np.log2(self.init_dim))
        self.data_dim = 2 # Is this a 2D data or 3D data

        self.b_res = np.exp((np.log(res_max) - np.log(res_min)) / (self.levels))
        self.b_token = np.exp((np.log(512) - np.log(table_num)) / (self.levels))

        self.token_num = token_num = 512 if not shrink_down else int(2 ** table_size_log2)
        token_dim = self.init_dim if not shrink_down else table_num
        # NOTE: we forcefully set 
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(token_num,
                                                token_dim,
                                                res_min=res_min,
                                                res_max=res_max,
                                                split_n=None if not shrink_down else table_num,
                                               ))
        self._build_layers()
        dprint('Finished building hash table generator.', color='g')

    @property
    def s_num(self) -> int:
        s_num = self.levels + 3
        if self.output_skip:
            s_num = s_num * 2
        return s_num

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
            block_num = 1 if not self.shrink_down else 1
            # Sample size and sample res
            sample_size = min(dim_now // 2, 128)
            sample_res = int(np.floor(self.res_min * self.b_res ** i))
            # If shrink down the table_num (token_num) will change
            token_num_now = (int(np.ceil(512 / (self.b_token ** i))) if not self.shrink_down
                                else int(self.token_num / (2 ** i)))
            next_token_num = (int(np.ceil(512 / (self.b_token ** (i + 1)))) if
                              not self.shrink_down else None)
            if next_token_num is not None and next_token_num < self.table_num:
                next_token_num = self.table_num
            # Every transformer block has 2 transform layers
            transformer_block = StackedModulatedGridLinear(dim_now,
                                                           self.style_dim,
                                                           token_num_now,
                                                           layer_n=2,
                                                           activation=nn.ReLU,
                                                           add_pos_encodings=False,
                                                           inter_filter=self.inter_filter,
                                                           sample_res=sample_res,
                                                           next_token_num=next_token_num,
                                                           upsample=not self.shrink_down
                                                           )
            
            setattr(self, f'transformer_block_{i}', transformer_block)
            if self.output_skip:
                setattr(self, f'hashside_out_{i}',
                        HashSideOut(self.res_min,
                                    sample_res,
                                    token_num_now,
                                    self.style_dim
                                    ))

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
        out = None
        for i in range(self.levels+1):
            x = getattr(self, f'transformer_block_{i}')(x, next(s1_iter))
            if self.output_skip:
                x_out = getattr(self, f'hashside_out_{i}')(x, next(s1_iter))
                if out is not None: 
                    out = F.interpolate(out, (x_out.size(2), x_out.size(3)),
                                        mode='bilinear') + x_out
                else:
                    out = x_out

            # Upscale or output
            if i != self.levels:
                # Upscale as concat
                if not self.shrink_down:
                    x = torch.cat((x, x), dim=-1)
                else:
                    x = x.reshape(b, x.size(1) // 2, 2, -1)
                    x = x.reshape(b, x.size(1), -1)
            elif not self.output_skip:
                out = x

        if not self.output_skip:
            out = out.reshape(out.shape[0], out.shape[1],
                                    out.shape[2] // self.F,
                                    self.F).contiguous()

        return out



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
