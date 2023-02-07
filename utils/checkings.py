"""Some checking functions"""
from typing import List

import torch

__all__ = ['CHECK_DIM', 'CHECK_DIM_VALUE']

def CHECK_DIM(name: str, x: torch.Tensor, dim_num: int):
    if x.dim() != dim_num:
        raise RuntimeError(f'Required input {name} has dim {dim_num} but got'
                           f' {x.dim()}.')

def CHECK_DIM_VALUE(name: str, x: torch.Tensor, dim: int, v: List[int]):
    if x.shape[dim] not in v:
        raise RuntimeError(f'Required input {name} dim {dim} has value {v}'
                           f' but got {x.shape[dim]}')
