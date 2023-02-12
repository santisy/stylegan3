from torch.utils.cpp_extension import load
from .cuda_coord_linear_interp import HashTableRetrieve

get_hash_up_mat = load('get_hash_up_matrix',
                        sources=['hash_retrieve_module/get_hash_interp_matrix.cpp'])

__all__ = ['HashTableRetrieve', 'get_hash_up_mat']