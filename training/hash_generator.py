"""The Generator part, assembling differnent parts together."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '.')
from torch_utils import persistence

from hash_encoding import HashTableGenerator
from hash_retrieve_module import HashTableRetrieve


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
        """
        super().__init__()
        self.res_min = res_min
        self.res_max = res_max
        self.img_size = img_size

        # Record for compability
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.hash_table_generator = HashTableGenerator(
                                        table_num=table_num,
                                        table_size_log2=table_size_log2,
                                        z_dim=z_dim,
                                        res_min=res_min,
                                        res_max=res_max,
                                        head_dim=head_dim)

        #TODO: substitue this to mini-mlp?
        # Mini linear for resolve hash collision
        self.mini_linear = nn.Sequential(
            nn.Linear(table_num * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_out_dim),
            nn.Tanh()
        )

    def _sample_coords(self, b) -> torch.Tensor:
        """
            Returns:
                coords: B x N x (2 or 3)
        """
        # 2D sampling case
        c = torch.arange(self.img_size) + 0.5
        x, y = torch.meshgrid(c, c)
        coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2)
        coords = coords.repeat(b, 1, 1) / self.img_size

        return coords

    def _render(self, x) -> torch.Tensor:
        """
            Render the sampled and went through linear results
            In 2D, it is just reshaping.

            Args:
                x: B x N x OUT_DIM
        """
        return  x.reshape(x.shape[0],
                            self.img_size,
                            self.img_size,
                            3).permute(0, 3, 1, 2)


    def forward(self, z: torch.Tensor, c=None, **kwargs):
        """
            Args:
                z: B x Z_DIM
        """

        ## Getting hash tables
        # The output size of the hash_tables is 
        #   B x H_N (table_num) x H_S (2 ** table_size_log2 // 2) x 2
        hash_tables = self.hash_table_generator(z)


        ## Retrieve from hash tables according to coordinates
        # The output size of retrieved feature is
        #   B x N x (table_num * F), value range [0, 1]
        coords = self._sample_coords(z.shape[0]).to(hash_tables.dtype
                                               ).to(hash_tables.device)
        hash_retrieved_feature = HashTableRetrieve.apply(hash_tables,
                                                         coords,
                                                         self.res_min,
                                                         self.res_max)

        ## Go through small MLPs
        mlp_out = self.mini_linear(hash_retrieved_feature) # B x N x OUT_DIM

        ## Rendering
        out = self._render(mlp_out)

        return out
