import torch
import torch.nn as nn
import torch.nn.functional as F
from training.hash_autoencoder_generator import HashAutoGenerator
from utils.utils import sample_coords

__all__ = ['decode']

def decode_nc(G, nc):
    """
        Args:
            G: The encoder decoder network.
            nc: neural coordinates
    """
    def encode(self,
            img: torch.Tensor,
            key_codes = None,
            no_noise_perturb=False,
            **kwargs):
        mu, log_var = None, None
        if key_codes is not None:
            feat_coords = key_codes
        elif not self.use_kl_reg:
            feat_coords = self.img_encoder(img) # B x F_C_C x W x H
        else:
            mu, log_var = self.img_encoder(img)
            log_var = torch.clamp(log_var, -30.0, 20.0)
            std = torch.exp(0.5 * log_var)
            feat_coords = F.sigmoid(torch.randn_like(std) * std + mu)

        if self.noise_perturb and not no_noise_perturb and key_codes is None:
            feat_coords = torch.clip(
                    feat_coords  +
                    torch.randn_like(feat_coords) * self.noise_perturb_sigma,
                    0, 1)

        return feat_coords, mu, log_var

    try:
        # rebind the encode to fix the encode bug
        G.encode = encode.__get__(G, HashAutoGenerator)
        out = G.synthesis(None, None, None, key_codes=nc)
    except:
        print("Not using class self `synthesis` function")
        b = nc.size(0)
        feat_coords = nc

        # Split the coordinates
        if G.expand_dim > 0:
            feat_coords = feat_coords.repeat_interleave(
                G.expand_dim // G.feat_coord_dim, dim=1)
        # Split the coordinates
        if not getattr(G, "circular_reuse", False):
            feat_coords_tuple = feat_coords.chunk(G.hash_encoder_num, dim=1)
        else:
            feat_coords_tuple = [feat_coords[:, g_idx] for g_idx in G.group_of_key_codes]
        coords = sample_coords(b, G.init_res).to(nc.device) # [0, 1], shape (B x N) x (2 or 3)
        coords = coords.reshape(-1, 2)

        feat_collect = []
        modulation_s_collect = []
        for i in range(G.hash_encoder_num):
            feat_coords_now = feat_coords_tuple[i]
            w = feat_coords_now.shape[2]
            repeat_ratio = G.init_res // w
            # Repeat the spatial
            feat_coords_now = feat_coords_now.repeat_interleave(repeat_ratio, dim=2)
            feat_coords_now = feat_coords_now.repeat_interleave(repeat_ratio, dim=3)
            feat_coords_now = feat_coords_now.permute(0, 2, 3, 1
                                                        ).reshape(
                                                        b*G.init_res*G.init_res, -1)
            input_coords = torch.cat((coords, feat_coords_now), dim=1)

            out1, out2 = G.hash_encoder_list[i](input_coords,
                                                    None,
                                                    None,
                                                    b=b)
            feat_collect.append(out1)
            modulation_s_collect.append(out2)
        modulation_s = torch.cat(modulation_s_collect, dim=-1) 
        modulation_s = modulation_s.unsqueeze(dim=1).repeat((1, G.synthesis_network.num_ws, 1))
        feats = torch.cat(feat_collect, dim=-1)
        feats = feats.reshape(b, G.init_res, G.init_res, G.init_dim)
        feats = feats.permute(0, 3, 1, 2)
        out = G.synthesis_network(modulation_s, feats)
    
    return out
