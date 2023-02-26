"""The Generator part, assembling differnent parts together."""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from torch_utils import persistence

from hash_encoding import HashTableGenerator
from hash_encoding.layers import ModulatedLinear
from hash_encoding.modules import StackedModulatedMLP
from hash_retrieve_module import HashTableRetrieve
from training.networks_stylegan2 import FullyConnectedLayer
from utils.utils import sinuous_pos_encode
from utils.utils import sample_coords
from utils.utils import render


@persistence.persistent_class
class HashGenerator(nn.Module):
    def __init__(self,
                 table_num: int,
                 table_size_log2: int,
                 z_dim: int,
                 res_min: int,
                 res_max: int,
                 head_dim: int=64,
                 mlp_hidden: int=64,
                 mlp_out_dim: int=3,
                 img_size: int=256,
                 c_dim: int=1,
                 init_dim: int=32,
                 style_dim: int=256,
                 use_layer_norm: bool=False,
                 modulated_mini_linear: bool=True,
                 more_layer_norm: bool=False,
                 fixed_random: bool=True,
                 linear_up: bool=True,
                 s_avg_beta: float=0.998,
                 output_skip: bool=True,
                 map_depth: int=2,
                 shuffle_input: bool=False,
                 spatial_atten: bool=False,
                 two_style_code: bool=False,
                 tokenwise_linear: bool=False,
                 shrink_down: bool=False,
                 no_norm_layer: bool=False,
                 **kwargs
                 ):
        """
            Overall generator interface. Generate the hash tables and 
                retrieve features according to coordinates.
            Args:
                table_num (int): how many tables (scales) we want to use.
                table_size_log2 (int): the final size will be 2 ** table_size_log2
                z_dim (int): the dimension of the latent vector
                res_max (int): maximum resolution (Usually finally rendered size.)
                res_min (int): minimum resolution
                head_dim (int): head_dim, head dimension for a single unit.
                    (default: 64)
                c_dim (int): conditional vector dimension.
                init_dim (int): initial size of the hash map dimension. 
                    (default: 32)
                style_dim (int): The mapped style code dimension. 
                    (default: 256)
                use_layer_norm (bool): Whether to use layer normalization in 
                    transformer. (default: False)
                modulated_mini_linear (bool): Modulated mini-linear layer 
                    following the generated hash tables. (default: True)
                more_layer_norm (bool): More layer normalizations in linear
                    layer. (default: False)
                fixed_random (bool): The fixed weight is randomized or not.
                linear_up (bool): If use linear up or not.
                w_avg_beta (float): Decay for tracking moving average of S
                    during training.
                output_skip (bool): If use output skip. (default: True)
                map_depth (bool): Mappig network depth. (default: 2)
                shuffle_input (bool): shuffle input of each block according to 
                    indices (default: False)
                spatial_atten (bool): Spatial attention or not. (default: False)
                tokenwise_linear (bool): If we use tokenwise linear or not.
                    (default: False)
                two_style_code (bool): Whether to use two separate style codes
                    for horizontal and vertical modulation.
                shrink_down (bool): Shrink down or not of the arch.
                    (default: False)
                no_norm_layer (bool): No normalization layer.
        """
        super().__init__()
        self.res_min = res_min
        self.res_max = res_max
        self.img_size = img_size
        self.modulated_mini_linear = modulated_mini_linear
        self.s_avg_beta = s_avg_beta
        self.shrink_down = shrink_down
        self.output_skip = output_skip
        self.table_num = table_num

        # Record for compability
        self.z_dim = z_dim
        self.c_dim = c_dim

        # The s_mapping network
        assert z_dim == style_dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(style_dim, 
                                                               nhead=4,
                                                               dim_feedforward=style_dim*2,
                                                               batch_first=True)
        self.s_mapping = nn.TransformerEncoder(transformer_encoder_layer,
                                               map_depth)
        self.register_buffer('pos_encoding',
                             sinuous_pos_encode(table_num, z_dim))

        # The hash table generator based on style
        self.hash_table_generator = HashTableGenerator(
                                        z_dim=z_dim,
                                        table_num=table_num,
                                        table_size_log2=table_size_log2,
                                        res_min=res_min,
                                        res_max=res_max,
                                        head_dim=head_dim,
                                        init_dim=init_dim,
                                        style_dim=style_dim,
                                        use_layer_norm=use_layer_norm,
                                        fixed_random=fixed_random,
                                        linear_up=linear_up,
                                        output_skip=output_skip,
                                        shuffle_input=shuffle_input,
                                        spatial_atten=spatial_atten,
                                        tokenwise_linear=tokenwise_linear,
                                        no_norm_layer=no_norm_layer,
                                        shrink_down =shrink_down
                                        )

        #TODO: substitue this to mini-mlp?
        # Mini linear for resolve hash collision
        self.mini_linear = nn.Sequential(nn.Linear(self.table_num * 2, 64),
                                         nn.LeakyReLU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 64),
                                         nn.LeakyReLU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 64),
                                         nn.LeakyReLU(),
                                         nn.LayerNorm(64),
                                         nn.Linear(64, 3),
                                         nn.Tanh()
                                         ) 


        self.register_buffer('s_avg', torch.zeros([style_dim]))

        self.b_level_img_size = np.exp((np.log(res_max) - np.log(res_min)) / (self.get_total_level_num - 1))

    def get_current_level_res_size(self, level: int) -> int:
        return int(self.res_min * self.b_level_img_size ** max(level, 0))

    @property
    def get_total_level_num(self) -> int:
        return self.hash_table_generator.levels + 1

    def _sample_coords(self, b, img_size: int=None) -> torch.Tensor:
        """
            Returns:
                coords: B x N x (2 or 3)
                img_size (int): Possible different img size during training
        """
        # 2D sampling case
        img_size = self.img_size if img_size is None else img_size
        coords = sample_coords(b, img_size)
        return coords

    def _render(self, x, img_size: int=None) -> torch.Tensor:
        """
            Render the sampled and went through linear results
            In 2D, it is just reshaping.

            Args:
                x: B x N x OUT_DIM
                img_size (int): Possible different img size during training
        """
        img_size = self.img_size if img_size is None else img_size
        return render(x, img_size)

    def mapping(self, z, c=None, update_emas=False, truncation_psi=1, **kwargs) -> torch.Tensor:
        # Main MLPs
        b = z.size(0)
        z = z.unsqueeze(dim=1) + self.pos_encoding.repeat(b, 1, 1)
        s = self.s_mapping(z)
        
        return s # B x N x S_DIM

    def _hash_and_render_out(self, hash_tables, sample_size=None):
        # Retrieve from hash tables according to coordinates
        # The output size of retrieved feature is
        #   B x N x (table_num * F), value range [0, 1]
        coords = self._sample_coords(hash_tables.size(0),
                                     img_size=sample_size).to(hash_tables.dtype
                                                         ).to(hash_tables.device)
        hash_retrieved_feature = HashTableRetrieve.apply(hash_tables,
                                                         coords,
                                                         self.res_min,
                                                         self.res_max)
        mlp_out = self.mini_linear(hash_retrieved_feature) # B x N x OUT_DIM
        ## Rendering
        out = self._render(mlp_out, sample_size)
        return out

    def synthesis(self,
                  s: torch.Tensor,
                  z: torch.Tensor,
                  update_emas: bool=False,
                  sample_size: int=None) -> torch.Tensor:
        """
            Args:
                s: style tokens
                out_level (int): If not None, the level will return at out_level
                    and only train out_level + 1
                linear_fuse_ration (float): If None, then we would fuse the 
                    lower level output with larger level. Fading in.
        """
        ## Getting hash tables
        # The output size of the hash_tables is 
        #   B x H_N (table_num) x H_S (2 ** table_size_log2 // 2) x 2
        out = self.hash_table_generator(s, z)
        # If output skip, then no need to hash out again.
        if not self.output_skip:
            out = self._hash_and_render_out(out, sample_size=sample_size)
        return out


    def forward(self, z: torch.Tensor, c=None, update_emas=False, **kwargs):
        """
            Args:
                z: B x Z_DIM
        """
        # The style mapping
        s = self.mapping(z, c)
        img = self.synthesis(s, z, update_emas=update_emas)
        return img


if __name__ == '__main__':
    h = HashGenerator(table_num=16, table_size_log2=13,
                      z_dim=128, res_min=16, res_max=256, init_dim=32).cuda()
    input_var = torch.randn(4, 128).cuda()
    out_var = h(input_var)
    print(out_var.shape)
