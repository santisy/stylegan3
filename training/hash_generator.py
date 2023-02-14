"""The Generator part, assembling differnent parts together."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '.')
from torch_utils import persistence

from hash_encoding import HashTableGenerator
from hash_encoding.layers import ModulatedLinear
from hash_encoding.modules import StackedModulatedMLP
from hash_retrieve_module import HashTableRetrieve
from training.networks_stylegan2 import FullyConnectedLayer
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
        """
        super().__init__()
        self.res_min = res_min
        self.res_max = res_max
        self.img_size = img_size
        self.modulated_mini_linear = modulated_mini_linear
        self.s_avg_beta = s_avg_beta

        # Record for compability
        self.z_dim = z_dim
        self.c_dim = c_dim

        # If use more normalization layer
        norm_layer = nn.LayerNorm if more_layer_norm else nn.Identity

        # The s_mapping network
        self.s_mapping = nn.Sequential(
            FullyConnectedLayer(z_dim, style_dim, activation='lrelu'),
            norm_layer(style_dim),
            FullyConnectedLayer(style_dim, style_dim, activation='lrelu'),
            norm_layer(style_dim),
            FullyConnectedLayer(style_dim, style_dim, activation='lrelu'),
            norm_layer(style_dim)
        )

        # The hash table generator based on style
        self.hash_table_generator = HashTableGenerator(
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
                                        output_skip=output_skip)

        #TODO: substitue this to mini-mlp?
        # Mini linear for resolve hash collision
        self.mini_linear = StackedModulatedMLP(table_num * 2,
                                            mlp_hidden,
                                            mlp_out_dim,
                                            style_dim,
                                            3,
                                            norm_layer=norm_layer)


        self.register_buffer('s_avg', torch.zeros([style_dim]))

    def _sample_coords(self, b) -> torch.Tensor:
        """
            Returns:
                coords: B x N x (2 or 3)
        """
        # 2D sampling case
        coords = sample_coords(b, self.img_size)
        return coords

    def _render(self, x) -> torch.Tensor:
        """
            Render the sampled and went through linear results
            In 2D, it is just reshaping.

            Args:
                x: B x N x OUT_DIM
        """
        return render(x, self.img_size)

    def mapping(self, z, c=None, update_emas=False, truncation_psi=1, **kwargs) -> torch.Tensor:
        # Main MLPs
        s = self.s_mapping(z)

        # Update EMAs
        if update_emas:
            self.s_avg.copy_(s.detach().mean(dim=0).lerp(self.s_avg, self.s_avg_beta))

        # Apply truncation
        if truncation_psi != 1:
            s = self.s_avg.lerp(s, truncation_psi)

        return s.unsqueeze(dim=1).repeat([1, self.hash_table_generator.levels + 2, 1])

    def synthesis(self, s: torch.Tensor, update_emas=False) -> torch.Tensor:
        """
            Args:
                s: is the repeated fashion, for the stylemixing regularization
        """
        ## Getting hash tables
        # The output size of the hash_tables is 
        #   B x H_N (table_num) x H_S (2 ** table_size_log2 // 2) x 2
        hash_tables = self.hash_table_generator(s)


        # Retrieve from hash tables according to coordinates
        # The output size of retrieved feature is
        #   B x N x (table_num * F), value range [0, 1]
        coords = self._sample_coords(s.shape[0]).to(hash_tables.dtype
                                               ).to(hash_tables.device)
        hash_retrieved_feature = HashTableRetrieve.apply(hash_tables,
                                                         coords,
                                                         self.res_min,
                                                         self.res_max)

        # Go through small MLPs
        if self.modulated_mini_linear:
            mlp_out = self.mini_linear(hash_retrieved_feature, s[:, -1]) # B x N x OUT_DIM
        else:
            mlp_out = self.mini_linear(hash_retrieved_feature)

        ## Rendering
        out = self._render(mlp_out)

        return out


    def forward(self, z: torch.Tensor, c=None, update_emas=False, **kwargs):
        """
            Args:
                z: B x Z_DIM
        """
        # The style mapping
        s = self.mapping(z, c)
        img = self.synthesis(s, update_emas=update_emas)
        return img


if __name__ == '__main__':
    h = HashGenerator(table_num=16, table_size_log2=13,
                      z_dim=128, res_min=16, res_max=256, init_dim=32).cuda()
    input_var = torch.randn(4, 128).cuda()
    out_var = h(input_var)
    print(out_var.shape)
