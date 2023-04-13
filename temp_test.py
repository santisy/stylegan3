import torch
from utils.utils import sample_coords
from utils.utils import pos_encodings
from hash_retrieve_module import HashTableRecon

res_min = 16
table_dim = 512
table_num = 16
res = 256
coords = sample_coords(1, res).cuda()
encodings = pos_encodings(res, table_num//2).cuda()

out = HashTableRecon.apply(encodings, coords, 512, 16, 256)

print(out.shape)
