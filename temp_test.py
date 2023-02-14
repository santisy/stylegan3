import torch
from hash_encoding.modules import MultiHeadAttention

mha = MultiHeadAttention(64, 4, None, 128, hidden_dim=128, out_dim=64)
input_var = torch.randn(4, 64, 64)
style_vector = torch.randn(4, 128)

out = mha(input_var, style_vector)
print(out.shape)
