"""Hash retrieve generative training"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import persistence
from gridencoder import GridEncoder
from training.networks_stylegan2 import SynthesisNetworkFromHash
from training.movq_module import MOVQDecoder
from training.encoder import Encoder
from utils.utils import itertools_combinations
from utils.utils import pos_encodings
from utils.utils import sample_coords
from utils.dist_utils import dprint

__all__ = ['HashAutoGenerator']

@persistence.persistent_class
class HashAutoGenerator(nn.Module):
    def __init__(self,
                 table_num: int,
                 table_size_log2: int,
                 z_dim: int,
                 res_min: int,
                 res_max: int,
                 level_dim: int=4,
                 map_depth: int=2,
                 feat_coord_dim: int=16,
                 init_res: int=32,
                 init_dim: int=256,
                 c_dim: int=1,
                 dummy_hash_table: bool=False,
                 tile_coord: bool=False,
                 discrete_all: bool=False,
                 mini_linear_n_layers: int=3,
                 style_dim: int=512,
                 num_downsamples: int=5,
                 feat_coord_dim_per_table: int=2,
                 additional_decoder_conv: bool=False,
                 noise_perturb: bool=False,
                 noise_perturb_sigma: float=-1.0,
                 use_kl_reg: bool=False,
                 hash_res_ratio: int=1,
                 expand_dim: int=-1,
                 attn_resolutions=None,
                 fused_spatial=False,
                 vq_decoder=False,
                 circular_reuse: bool=False,
                 larger_decoder: bool=False,
                 encoder_ch: int=32,
                 movq_decoder: bool=False,
                 encoder_resnet_num: int=4,
                 no_concat_coord: bool=False,
                 hash_resolution: int=-1,
                 **kwargs):
        """
            Args:
                table_num (int): how many tables (scales) we want to use.
                table_size_log2 (int): the final size will be 2 ** table_size_log2
                z_dim (int): the dimension of the latent vector
                res_max (int): maximum resolution (Usually finally rendered size.)
                res_min (int): minimum resolution
                level_dim (int): Each entry feature dimension in hash table.
                    (default: 4)
                map_depth (bool): Mappig network depth. (default: 2)
                feat_coord_dim (int): style to hyperspace coordinate 
                    dimension. (default: 16)
                init_res (int): initial resolution retrieved from hash tables.
                    (default: 32)
                c_dim (int): conditional vector dimension. (default: 1)
                int_dim (int): result input to the convolution network. 
                    (default: 256)
                dummy_hash_table (bool): dummy hash table output for ablation
                    study. (default: False)
                tile_coord (bool): tile coord with spatial coordinates or not.
                    (default: False)
                discrete_all (bool): If true discrete the style code also.
                    (default: False)
                mini_linear_n_layers (int): Mini-linear n layers. (default: 3)
                one_hash_group (bool): Only one hash map, which outputs the tensor
                    chunk and style code.
                non_style_decoder (bool): The decoder is non-style fashion.
                    (default: False)
                context_coordinates (bool): Patch-wise coordinates.
                    (default: False)
                additional_decoder_conv (bool): Additional decoder convolution.
                    (default: False)
                noise_perturb (bool): Noise perturbation on neural coordinates or
                    not (default: False)
                noise_perturb_sigma (bool): The sigma value to perturb the 
                    neural coordinates. If -1.0, then we would calculate the 
                    sigma according to the .
                    (default: -1)
                use_kl_reg (bool): Use KL regularization or not.
                hash_res_ratio (bool): Hash maximum resolution ratio to the init res.
                    (default: 1)
                expand_dim (int): expand dimsion of neural indexing channel-wise.
                    (default: -1)
                attn_resolutions (List): attn resolutions in encoder.
                    (default: False)
                fused_spatial (bool): Fused spatial. y in hash and x in modulation.
                    (default: False)
                vq_decoder (bool): Use vq decoder? (default: False)
                circular_reuse (bool): Circular reuse the code with 
                    feat_coord_dim_per_table > 1.
                    (default: False)
                larger_decoder (bool): Even larger and more complicated decoder.
                    (default: False)
                encoder_ch (int): encoder unit channel size
                    (default: 32)
                movq_decoder (bool): MOVQ deocder or not.
                    (default: False)
                encoder_resnet_num (int): the number of encoder blocks
                    (default: 4)
                no_concat_coord (bool): Do not concat coordinates
                    (default: False)
        """

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.init_dim = init_dim
        self.init_res = init_res
        self.discrete_all = discrete_all
        self.feat_coord_dim = feat_coord_dim
        self.noise_perturb = noise_perturb
        self.use_kl_reg = use_kl_reg
        self.expand_dim = expand_dim
        self.fused_spatial = fused_spatial
        self.vq_decoder = vq_decoder
        self.circular_reuse = circular_reuse
        self.feat_coord_dim_per_table = feat_coord_dim_per_table
        self.movq_decoder = movq_decoder
        self.no_concat_coord = no_concat_coord

        if no_concat_coord:
            spatial_coord_dim = 0
        elif fused_spatial:
            spatial_coord_dim = 1
        else:
            spatial_coord_dim = 2

        if noise_perturb_sigma > 0:
            self.noise_perturb_sigma = noise_perturb_sigma
        else:
            # Automatically calculate the noise pterturb sigma according to 
            #   the 2*std=half of the resolution
            self.noise_perturb_sigma = 1. / (init_res * hash_res_ratio * 2 * 2)

        if expand_dim > 0:
            self.hash_encoder_num = expand_dim // feat_coord_dim_per_table
        else:
            if not circular_reuse:
                self.hash_encoder_num = feat_coord_dim // feat_coord_dim_per_table
            else:
                self.hash_encoder_num = math.comb(feat_coord_dim,
                                                  feat_coord_dim_per_table)
                group_of_key_codes = torch.from_numpy(itertools_combinations(
                    np.arange(feat_coord_dim), feat_coord_dim_per_table)).long()
                self.register_buffer('group_of_key_codes', group_of_key_codes)

        self.img_encoder = Encoder(feat_coord_dim=feat_coord_dim,
                                   ch=encoder_ch,
                                   max_ch=512,
                                   num_res_blocks=encoder_resnet_num,
                                   num_downsamples=num_downsamples,
                                   resolution=res_max,
                                   use_kl_reg=use_kl_reg,
                                   attn_resolutions=attn_resolutions
                                   )

        self.hash_encoder_list = nn.ModuleList()
        for _ in range(self.hash_encoder_num):
            self.hash_encoder_list.append(GridEncoder(input_dim=feat_coord_dim_per_table + spatial_coord_dim,
                                            num_levels=table_num,
                                            level_dim=level_dim,
                                            base_resolution=res_min,
                                            log2_hashmap_size=table_size_log2,
                                            desired_resolution=init_res * hash_res_ratio if hash_resolution < 0 else hash_resolution,
                                            feat_coord_dim=feat_coord_dim_per_table,
                                            out_dim=init_dim // self.hash_encoder_num,
                                            dummy_hash_table=dummy_hash_table,
                                            style_dim=style_dim,
                                            out_dim2=style_dim // self.hash_encoder_num,
                                            tile_coord=tile_coord,
                                            mini_linear_n_layers=mini_linear_n_layers,
                                            direct_coord_input=True,
                                            no_modulated_linear=False,
                                            one_hash_group=True,
                                            fused_spatial=fused_spatial
                                            ))
        dprint('Number of groups of hash tables is'
               f' {len(self.hash_encoder_list)}', color='g')

        if vq_decoder:
            resample_filter = None
        else:
            resample_filter = [1, 3, 3, 1]

        if not movq_decoder:
            self.synthesis_network = SynthesisNetworkFromHash(style_dim,
                                                              res_max,
                                                              3,
                                                              channel_base=max(32768, self.init_dim * self.init_res),
                                                              channel_max=self.init_dim,
                                                              num_fp16_res=0,
                                                              init_res=init_res,
                                                              additional_decoder_conv=additional_decoder_conv,
                                                              resample_filter=resample_filter,
                                                              larger_decoder=larger_decoder
                                                              )
        else:
            self.synthesis_network = MOVQDecoder(init_dim,
                                                 init_dim,
                                                 3,
                                                 init_res,
                                                 res_max,
                                                 num_res_blocks=2,
                                                 attn_resolutions=None)
            self.register_buffer('const_fourier_input',
                                 pos_encodings(init_res, init_dim // 4).reshape(
                                    1, init_res, init_res, init_dim).permute(0, 3, 1, 2)
                                 )

        dprint('Finished building hash table generator.', color='g')


    def mapping(self, z, c=None, update_emas=False, truncation_psi=1, **kwargs) -> torch.Tensor:
        """Legacy placeholder"""

        return None, None

    def synthesis(self,
                  s: torch.Tensor,
                  s2: torch.Tensor,
                  img: torch.Tensor,
                  update_emas: bool=False,
                  sample_size: int=None,
                  return_kl_terms: bool=False,
                  ) -> torch.Tensor:
        """
            Args:
                s: is the repeated fashion, for the stylemixing regularization
                out_level (int): If not None, the level will return at out_level
                    and only train out_level + 1
                linear_fuse_ration (float): If None, then we would fuse the 
                    lower level output with larger level. Fading in.
        """
        b = img.size(0)

        if not self.use_kl_reg:
            feat_coords = self.img_encoder(img) # B x F_C_C x W x H
        else:
            mu, log_var = self.img_encoder(img)
            std = torch.exp(0.5 * log_var)
            feat_coords = F.sigmoid(torch.randn_like(std) * std + mu)

        if self.noise_perturb:
            feat_coords = torch.clip(
                    feat_coords  +
                    torch.randn_like(feat_coords) * self.noise_perturb_sigma,
                    0, 1)

        # Split the coordinates
        if self.expand_dim > 0:
            feat_coords = feat_coords.repeat_interleave(self.expand_dim // self.feat_coord_dim, dim=1)
        if not self.circular_reuse:
            feat_coords_tuple = feat_coords.chunk(self.hash_encoder_num, dim=1)
        else:
            feat_coords_tuple = [feat_coords[:, g_idx] for g_idx in self.group_of_key_codes]
        coords = sample_coords(b, self.init_res).to(img.device) # [0, 1], shape (B x N) x (2 or 3)
        coords = coords.reshape(-1, 2)
        if self.fused_spatial:
            # This is the modulation coordiates
            mod_coords = coords[:, 0].unsqueeze(dim=1)
            # This is the concatenation coords
            coords = coords[:, 1].unsqueeze(dim=1)
        else:
            mod_coords = None

        feat_collect = []
        modulation_s_collect = []
        for i in range(self.hash_encoder_num):
            feat_coords_now = feat_coords_tuple[i]
            w = feat_coords_now.shape[2]
            repeat_ratio = self.init_res // w
            # Repeat the spatial
            if not self.no_concat_coord:
                feat_coords_now = feat_coords_now.repeat_interleave(repeat_ratio, dim=2)
                feat_coords_now = feat_coords_now.repeat_interleave(repeat_ratio, dim=3)
                feat_coords_now = feat_coords_now.permute(0, 2, 3, 1
                                                        ).reshape(
                                                        b*self.init_res*self.init_res, -1)
                input_coords = torch.cat((coords, feat_coords_now), dim=1)
            else:
                input_coords = feat_coords_now

            out1, out2 = self.hash_encoder_list[i](input_coords,
                                                   None,
                                                   None,
                                                   b=b,
                                                   mod_coords=mod_coords)
            feat_collect.append(out1)
            modulation_s_collect.append(out2)
        feats = torch.cat(feat_collect, dim=-1)
        feats = feats.reshape(b, self.init_res, self.init_res, self.init_dim)
        feats = feats.permute(0, 3, 1, 2)
        if not self.movq_decoder:
            modulation_s = torch.cat(modulation_s_collect, dim=-1) 
            modulation_s = modulation_s.unsqueeze(dim=1).repeat((1, self.synthesis_network.num_ws, 1))
            out = self.synthesis_network(modulation_s, feats)
        else:
            out = self.synthesis_network(
                self.const_fourier_input.repeat(b, 1, 1, 1), feats)
        
        if not return_kl_terms:
            return out
        else:
            return out, mu, log_var


    def forward(self, img: torch.Tensor, c=None, update_emas=False, **kwargs):
        """
            Args:
                z: B x Z_DIM
        """
        # The style mapping
        s, s2 = self.mapping(None, c)
        img = self.synthesis(s, s2, img, update_emas=update_emas)
        return img
