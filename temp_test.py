import torch
from hash_encoding.modules import HashAttention

ha = HashAttention(table_num=16, res_min=16, res_max=256,
                   style_dim=256, sample_size=128, head_num=4).cuda()
input_var = torch.randn(2, 16, 256).cuda()
s = torch.randn(2, 256).cuda()

out = ha(input_var, s)
print(out.shape)
