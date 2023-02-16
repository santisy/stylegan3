import torch
from hash_encoding.layers import TokenWiseModulatedLinear

twml = TokenWiseModulatedLinear(128, 128, 16, 128).cuda()
input_var = torch.randn(2, 16, 128).cuda()
s = torch.randn(2, 128).cuda()
out = twml(input_var, s)
print(out.shape)
