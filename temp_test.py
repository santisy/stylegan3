"""Test movq module correct or not"""
import torch
from training.movq_module import MOVQDecoder

decoder = MOVQDecoder(512, 512, 3, 64, 256).cuda()
const_pos_input = torch.randn(2, 512, 64, 64).cuda()
z_input = torch.randn(2, 512, 64, 64).cuda()

out = decoder(const_pos_input, z_input)
print(out.shape)
