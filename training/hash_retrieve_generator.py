"""Hash retrieve generative training"""

import torch
import torch.nn as nn

from torch_utils import persistence
from hash_encoding.other_networks import MappingNetwork
from gridencoder import GridEncoder
from training.networks_stylegan2 import SynthesisNetworkFromHash
from utils.utils import sample_coords
from utils.dist_utils import dprint

__all__ = ['HashRetrieveGenerator']

FEAT_COORD_DIM_PER_TABLE = 2

@persistence.persistent_class
class HashRetrieveGenerator(nn.Module):
    def __init__(self,
                 table_num: int,
                 table_size_log2: int,
                 z_dim: int,
                 res_min: int,
                 res_max: int,
                 style_dim: int=256,
                 level_dim: int=4,
                 map_depth: int=2,
                 feat_coord_dim: int=16,
                 init_res: int=64,
                 init_dim: int=256,
                 c_dim: int=1,
                 dummy_hash_table: bool=False,
                 **kwargs):
        """
            Args:
                table_num (int): how many tables (scales) we want to use.
                table_size_log2 (int): the final size will be 2 ** table_size_log2
                z_dim (int): the dimension of the latent vector
                res_max (int): maximum resolution (Usually finally rendered size.)
                res_min (int): minimum resolution
                style_dim (int): The mapped style code dimension. 
                    (default: 256)
                level_dim (int): Each entry feature dimension in hash table.
                    (default: 4)
                map_depth (bool): Mappig network depth. (default: 2)
                feat_coord_dim (int): style to hyperspace coordinate 
                    dimension. (default: 16)
                init_res (int): initial resolution retrieved from hash tables.
                    (default: 64)
                c_dim (int): conditional vector dimension. (default: 1)
                int_dim (int): result input to the convolution network. 
                    (default: 256)
                dummy_hash_table (bool): dummy hash table output for ablation
                    study. (default: False)
        """

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.hash_encoder_num = feat_coord_dim // FEAT_COORD_DIM_PER_TABLE
        self.init_dim = init_dim
        self.init_res = init_res
        self.style_dim = style_dim

        self.s_mapping = MappingNetwork(in_ch=z_dim,
                                        map_depth=map_depth,
                                        style_dim=style_dim,
                                        hidden_ch=style_dim,
                                        use_layer_norm=False,
                                        two_style_code=False)


        self.hash_encoder_list = nn.ModuleList()
        for _ in range(self.hash_encoder_num):
            self.hash_encoder_list.append(GridEncoder(input_dim=FEAT_COORD_DIM_PER_TABLE + 2, # Now only allows 16 + 2 or 16 + 3
                                            num_levels=table_num,
                                            level_dim=level_dim,
                                            base_resolution=res_min,
                                            log2_hashmap_size=table_size_log2,
                                            desired_resolution=init_res,
                                            style_dim=style_dim,  
                                            feat_coord_dim=FEAT_COORD_DIM_PER_TABLE,
                                            out_dim=init_dim // self.hash_encoder_num,
                                            dummy_hash_table=dummy_hash_table
                                            ))

        self.synthesis_network = SynthesisNetworkFromHash(style_dim,
                                                          res_max,
                                                          3,
                                                          channel_base=32768,
                                                          channel_max=self.init_dim,
                                                          num_fp16_res=2,
                                                          init_res=init_res,
                                                          )

        dprint('Finished building hash table generator.', color='g')


    def mapping(self, z, c=None, update_emas=False, truncation_psi=1, **kwargs) -> torch.Tensor:
        # Main MLPs
        s, _ = self.s_mapping(z)

        ## Update EMAs
        #if update_emas:
        #    self.s_avg.copy_(s.detach().mean(dim=0).lerp(self.s_avg, self.s_avg_beta))

        ## Apply truncation
        #if truncation_psi != 1:
        #    s = self.s_avg.lerp(s, truncation_psi)

        return s.unsqueeze(dim=1).repeat((1, self.synthesis_network.num_ws, 1)), None

    def synthesis(self,
                  s: torch.Tensor,
                  s2: torch.Tensor,
                  z: torch.Tensor,
                  update_emas: bool=False,
                  sample_size: int=None) -> torch.Tensor:
        """
            Args:
                s: is the repeated fashion, for the stylemixing regularization
                out_level (int): If not None, the level will return at out_level
                    and only train out_level + 1
                linear_fuse_ration (float): If None, then we would fuse the 
                    lower level output with larger level. Fading in.
        """
        ori_s = s
        b = s.size(0)
        coords = sample_coords(b, self.init_res).to(s.device) # [-1, 1], shape (B x N) x (2 or 3)
        s = s[:, 0]
        coords = coords.reshape(-1, 2)
        feat_collect = []
        for i in range(self.hash_encoder_num):
            feat_collect.append(self.hash_encoder_list[i](coords, s))
        feats = torch.cat(feat_collect, dim=-1)
        feats = feats.reshape(b, self.init_res, self.init_res, self.init_dim)
        feats = feats.permute(0, 3, 1, 2)
        out = self.synthesis_network(ori_s, feats)
        
        return out


    def forward(self, z: torch.Tensor, c=None, update_emas=False, **kwargs):
        """
            Args:
                z: B x Z_DIM
        """
        # The style mapping
        s, s2 = self.mapping(z, c)
        img = self.synthesis(s, s2, z, update_emas=update_emas)
        return img
