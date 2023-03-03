import torch
from hash_encoding.layers import ModulatedGridLinear

input_var = torch.randn(4, 16, 128).cuda()
style_code = torch.randn(4, 256).cuda()
mgl = ModulatedGridLinear(128, 128, 256, 16, add_pos_encodings=True).cuda()

out = mgl(input_var, style_code)
print(out.shape)
