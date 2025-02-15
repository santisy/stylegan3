"""
    Hash related operation pytorch interface
    Author: Dingdong Yang
    Contacts: dya62@sfu.ca
"""

import argparse
import time
from typing import List

import torch
from torch.utils.cpp_extension import load


hash_retrieve = load(name='hash_retrieve',
                        sources=['./hash_retrieve_module/cuda_coord_linear_interp.cpp',
                                 './hash_retrieve_module/cuda_coord_linear_kernel.cu'])

def CHECK_DIM(name: str, x: torch.Tensor, dim_num: int):
    if x.dim() != dim_num:
        raise RuntimeError(f'Required input {name} has dim {dim_num} but got'
                           f' {x.dim()}.')

def CHECK_DIM_VALUE(name: str, x: torch.Tensor, dim: int, v: List[int]):
    if x.shape[dim] not in v:
        raise RuntimeError(f'Required input {name} dim {dim} has value {v}'
                           f' but got {x.shape[dim]}')

class HashTableRetrieve(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                hash_tables: torch.Tensor,  # B x N_H x N_S x F
                coords: torch.Tensor,       # B x N x (3 or 2)
                res_min: int,                
                res_max: int):
        """
            Args:
                hash_tables (torch.Tensor, float or half): 
                    Size B x H_N x N_S x F. H_N means
                    number of levels of hash tables; N_S means the hash table
                    size, namely 'T' in the original instant-ngp paper.
                coords (torch.Tensor, float or half): N x N x (3 or 2), retrieve
                    2D or 3D coordinates. Values range in [0, 1]
                res_min (int): minimum hash table resolution.
                res_max (int): maximum hash table resolution.

            .. note::
                currently only support F == 2
        """

        # Checkings
        CHECK_DIM('hash_tables', hash_tables, 4)
        CHECK_DIM('coordiates', coords, 3)
        CHECK_DIM_VALUE('hash_tables', hash_tables, 3, [2,])
        CHECK_DIM_VALUE('coordinates', coords, 2, [2, 3])

        features, indices = hash_retrieve.retrieve_forward(hash_tables,
                                                           coords,
                                                           res_min,
                                                           res_max)
        ctx.res_min = res_min
        ctx.res_max = res_max
        ctx.table_size = hash_tables.size(2)
        ctx.save_for_backward(coords, indices)
        return features
    
    @staticmethod
    def backward(ctx, grad_out):
        coords, indices = ctx.saved_tensors
        grad_table = hash_retrieve.retrieve_backward(grad_out.contiguous(),
                                                     coords,
                                                     indices,
                                                     ctx.res_max,
                                                     ctx.res_min,
                                                     ctx.table_size)
        return grad_table, None, None, None
    

class HashTableRecon(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                feats: torch.Tensor, # B x N x (H_N x F)
                coords: torch.Tensor, # B x N x 2
                table_dim: int,
                res_min: int,
                res_max: int
                ):
        CHECK_DIM('feats', feats, 3)
        hash_table, weights, indices = hash_retrieve.recon_forward(feats,
                                                          coords,
                                                          table_dim,
                                                          res_min,
                                                          res_max)
        ctx.res_min = res_min
        ctx.res_max = res_max
        ctx.feat_size = feats.size(1)
        ctx.save_for_backward(weights, coords, indices)

        return hash_table

    @staticmethod
    def backward(ctx, grad_out):
        weights, coords, indices = ctx.saved_tensors
        grad_feats = hash_retrieve.recon_backward(grad_out.contiguous(),
                                                  coords,
                                                  weights,
                                                  indices,
                                                  ctx.res_min,
                                                  ctx.res_max,
                                                  ctx.feat_size)
        return grad_feats, None, None, None, None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timing', action='store_true',
                        help=('Get the average timing of the forward'
                                ' + backward runnning.'))
    parser.add_argument('-g', '--gradcheck', action='store_true',
                        help='Gradcheck flag of pytorch of this op.')
    args = parser.parse_args()

    # Const variable
    TEST_N = 1000
    TEST_RES = 256

    if args.timing:
        start = time.time()
        for _ in range(TEST_N):
            hash_tables = torch.randn(1, 12, 4096, 2).cuda().requires_grad_(True)
            coords = torch.rand(1, TEST_RES*TEST_RES, 3).cuda()
            features = HashTableRetrieve.apply(hash_tables, coords, 512, 16)
            features.sum().backward()
        end = time.time()
        print(f'Avg running time {(end - start)/ TEST_N * 1000:.2f} ms.')

    if args.gradcheck:
        hash_tables = torch.randn(1, 12, 4096, 2).cuda().requires_grad_(True).float()
        coords = torch.rand(1, TEST_RES*TEST_RES, 3).cuda().float()
        torch.autograd.gradcheck(HashTableRetrieve.apply,
                                (hash_tables, coords, 512, 16),
                                fast_mode=True)

    