"""This is the hash-table generator network based on """
import numpy as np

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from utils.utils import sinuous_pos_encode
from utils.dist_utils import dprint

class HashTableGenerator(nn.Module):
    def __init__(self,
                 table_num: int,
                 table_size_log2: int,
                 z_dim: int,
                 res_min: int,
                 res_max: int,
                 head_dim: int=64
                 ):
        """
            Args:
                table_num (int): how many tables (scales) we want to use.
                table_size_log2 (int): the final size will be 2 ** table_size_log2
                z_dim (int): the dimension of the latent vector
                res_max (int): maximum resolution
                res_min (int): minimum resolution
                head_dim (int): head_dim, head dimension for a single unit.
                    (default: 64)
        """
        super(HashTableGenerator, self).__init__()

        self.table_num = table_num
        self.table_size_log2 = table_size_log2
        self.res_max = res_max
        self.res_min = res_min
        self.head_dim = head_dim
        self.z_dim = z_dim
        self.F = 2 #NOTE: We only support entry size 2 now for CUDA programming reason

        b = np.exp((np.log(res_max) - np.log(res_min)) / (table_num - 1))
        res_list = res_min * b ** torch.arange(table_num)
        res_list = res_list.unsqueeze(dim=0) # 1 x H_N
        dprint(f'Hash table resolution list is {res_list}')
        self.input_dim = self.z_dim // table_num
        # The positional encodings' size is 1 x H_N x Z_DIM
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(table_num,
                                                self.input_dim))
        self._build_layers()

    def _build_layers(self) -> None:
        """This is where to put layer building."""
        # Upsample levels to go
        input_dim = self.input_dim
        self.levels = L = int(self.table_size_log2 - np.log2(input_dim))

        for i in range(L+1):
            dim_now = int(input_dim * 2 ** i)
            head_dim_now = min(dim_now, self.head_dim)
            # The dimension for a single head remains constant
            nhead_now = dim_now // head_dim_now

            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_now,
                                                       nhead=nhead_now)
            # Every transformer block has 2 transform layers
            transform_block = nn.TransformerEncoder(encoder_layer, num_layers=2)
            setattr(self, f'transformer_block_{i}', transform_block)
            if i != L:
                setattr(self, f'mlp_up_{i}', nn.Linear(dim_now, dim_now * 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
            Args:
                z (torch.Tensor): B x Z_DIM

            Return:
                hash_tables (torch.Tensor): B x H_N x H_S x (F=2)
        """
        #z = z.unsqueeze(dim=1) # N x 1 x Z_DIM
        b = z.shape[0]
        z = z.reshape(b, self.table_num, -1)

        x = tokenized_z = z + self.pos_encoding
        # Transformers following
        for i in range(self.levels+1):
            x = getattr(self, f'transformer_block_{i}')(x)
            if i == 0:
                out = x
            else:
                out = F.interpolate(out, scale_factor=2) + x
            if i != self.levels:
                x = getattr(self, f'mlp_up_{i}')(x)

        hash_tables = out.reshape(out.shape[0], out.shape[1],
                                  out.shape[2] // self.F, self.F)

        return hash_tables



if __name__ == '__main__':
    h_net = HashTableGenerator(table_num=16,
                               table_size_log2=13,
                               z_dim=128,
                               res_max=512,
                               res_min=16,
                               head_dim=64).cuda()

    z = torch.randn(4, 128).cuda()
    out = h_net(z)
    print(out.shape)
