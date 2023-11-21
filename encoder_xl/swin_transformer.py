"""Build SwinTransformer as the encoder"""
import math

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import SwinTransformer

class STEncoder(SwinTransformer):
    """
        SwinTransformer Encoder
    """
    def __init__(self,
                 out_dim,
                 patch_size=[4, 4],
                 depths=[6, 8, 12],
                 embed_dim=128,
                 num_heads=[4, 8, 12],
                 window_size=[7, 7],
                 stochastic_depth_prob=0.0,
                 num_downsamples=2,
                 **kwargs):

        depth_len = int(num_downsamples - math.log2(patch_size[0])  + 1)
        depths = depths[:depth_len]
        num_heads = num_heads[:depth_len]

        super().__init__(patch_size=patch_size,
                         stochastic_depth_prob=stochastic_depth_prob,
                         num_heads=num_heads,
                         depths=depths,
                         embed_dim=embed_dim,
                         window_size=window_size, 
                         **kwargs)

        in_dim = embed_dim * 2 ** (len(depths) - 1)
        self.out_conv = nn.Conv2d(in_dim, out_dim, 1, 1)


    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.out_conv(x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    import torch
    input_var = torch.randn(2, 3, 256, 256).cuda()
    encoder = STEncoder(out_dim=512).cuda()
    out = encoder(input_var)
    print(out.shape)
