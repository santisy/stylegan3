"""Some utilities"""
import os
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from hash_retrieve_module import HashTableRecon

__all__ = ['sinuous_pos_encode']


def sinuous_pos_encode(table_num: int, token_d: int,
                       res_min: int=None,
                       res_max: int=None,
                       split_n: int=None):
    """
        Args:
            table_num (int): table number for specify the index for each table
            token_d (int): frequency encoded dimension. The length of each input
                token.
            res_min (int): minimum resolution the table represent.
            res_max (int): maximum resolution the table represent.
            split_n (int): Is the table n already be splitted?

        Returns:
            positional encoded vector B x N x (dim x 2)
    """
    token_n = split_n if split_n is not None else table_num
    token_d_now = int(token_d * table_num / token_n)
    b = np.exp((np.log(res_max) - np.log(res_min)) / (token_n - 1))
    res_list = res_min * b ** np.arange(token_n)

    indices = torch.arange(token_n).unsqueeze(dim=1).float() / 10000.
    k = torch.zeros((token_d_now // 2, 2))
    k[:, 0] = (torch.arange(token_d_now // 2).float() * 2 / token_d_now)
    k[:, 1] = (torch.arange(token_d_now // 2).float() * 2 / token_d_now)
    k = k.flatten().unsqueeze(dim=0)

    temp_p = indices ** k
    temp_p = temp_p.unsqueeze(dim=0) # 1 x N (number of token) x L (token length)
    
    # Phase shift to generate both sin and cos
    shift_p = torch.zeros((token_d_now // 2, 2))
    shift_p[:, 1] = torch.pi / 2. * 3.0
    shift_p = shift_p.flatten()
    shift_p = shift_p.unsqueeze(dim=0).unsqueeze(dim=0)
    pos_encodings = torch.cos(temp_p + shift_p)

    pos_encodings = pos_encodings.reshape(1, table_num, token_d).detach()

    return pos_encodings


def get_hash_mask(res_max: int, res_min: int,
                  token_num: int, table_len: int) -> torch.Tensor:
    b = np.exp((np.log(res_max) - np.log(res_min))/(token_num - 1))
    hash_mask = torch.zeros((token_num, table_len))

    # Get 2D coords
    c = torch.arange(res_max) + 0.5
    x, y = torch.meshgrid(c, c, indexing='xy')
    # Normalize it to [0, 1]
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2) / res_max

    # 1, 2654435761
    # Iterate on each resolution
    for i in range(token_num):
        res_table = np.ceil(res_min * b ** i)
        current_coords = torch.floor(coords * res_table).long()
        for x, y in current_coords:
            hash_mask[i, ((x) ^ (y * 2654435761)) % table_len] = 1
            hash_mask[i, ((x + 1) ^ (y * 2654435761)) % table_len] = 1
            hash_mask[i, ((x) ^ ((y + 1) * 2654435761)) % table_len] = 1
            hash_mask[i, ((x + 1) ^ ((y + 1) * 2654435761)) % table_len] = 1

    return hash_mask.unsqueeze(dim=0)

def delete_file(file_path: str):
    # TODO: More to add if we involve HDFS file system
    # Remove the local path
    if os.path.isfile(file_path):
        os.remove(file_path)

def sample_coords(b: int, img_size: int,
                  sample_size: int=None,
                  single_batch: bool=False,
                  combine_coords: bool=False):
    """
        Args:
            b (int): batch_size
            img_size (int): image size
            combine_coords (bool): combine x,y coordinates to one dimension.
        Retrun:
            coords (torch.Tensor): B x N x (2 or 3), value range [0, 1]
    """
    # 2D sampling case
    c = torch.arange(img_size) + 0.5
    x, y = torch.meshgrid(c, c, indexing='xy')
    # Normalize it to [0, 1]
    if not combine_coords:
        coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2) / img_size
    else:
        coords = (x * img_size + y).reshape(1, -1, 1) / (img_size * img_size)

    if sample_size is not None:
        sampled_indices = torch.randperm(coords.shape[1])[:sample_size]
        coords = coords[:, sampled_indices, :]

    if not single_batch:
        coords = coords.repeat((b, 1, 1))

    return coords


def sample_local_coords(b: int, img_size: int, local_size: int):
    """
        Args:
            b (int): batch_size
            img_size (int): image size, output size
            local_size (int): 
        Retrun:
            coords (torch.Tensor): B x N x 1, value range [0, 1].
                Combine all local coordinates to a single dimension.
    """
    # 2D sampling case
    c = torch.arange(local_size) + 0.5
    x, y = torch.meshgrid(c, c, indexing='xy')
    # Normalize it to [0, 1]
    coords = (x * local_size + y) / (local_size * local_size)
    coords = coords.repeat((img_size // local_size, img_size // local_size))
    coords = coords.reshape(1, -1, 1)
    coords = coords.repeat((b, 1, 1))

    return coords


def pos_encodings(res: int, half_len: int):
    c = torch.arange(res) + 0.5
    x, y = torch.meshgrid(c, c, indexing='xy')
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2, 1).float()
    coords = coords / res * 2.0 - 1.0

    coeff = torch.pow(2, torch.arange(half_len)) * torch.pi
    coeff = coeff.unsqueeze(dim=0)
    x1 = torch.cos(coords[:, 0] * coeff)
    x2 = torch.sin(coords[:, 0] * coeff)
    y1 = torch.cos(coords[:, 1] * coeff)
    y2 = torch.sin(coords[:, 1] * coeff)

    total_pos_encodings = torch.cat((x1, y1, x2, y2), dim=-1).unsqueeze(dim=0)

    return total_pos_encodings


def hashed_positional_encodings(res: int, table_dim: int, table_num: int,
                                res_min: int=16):

    coords = sample_coords(1, res).cuda()
    encodings = pos_encodings(res, table_num//2).cuda()

    out = HashTableRecon.apply(encodings, coords, table_dim, res_min, res)
    out = out.reshape(1, table_num, -1)

    return out # 1 x table_num x table_dim x 2


def get_shuffle_table_indices(table_num: int, table_dim: int) -> torch.Tensor:
    """
        Args:
            table_num (int): number of tables
            table_dim (int): table dimension
        Returns:
            indices (torch.Tensor): Shuffle indices
    """
    indices_collect = []
    for _ in range(table_num):
        indices_collect.append(torch.randperm(table_dim))
    return torch.stack(indices_collect, dim=0).unsqueeze(dim=0)


def render(x: torch.Tensor, img_size: int):

    return  x.reshape(x.shape[0],
                        img_size,
                        img_size,
                        -1).permute(0, 3, 1, 2)

def smooth_ceiling(x, gamma=0.99):
    return x - 0.5 - torch.arctan(
                                    - gamma*torch.sin(2 * torch.pi * x)/(
                                    1 - gamma * torch.cos(2 * torch.pi * x))
                                ) / torch.pi


def pos_encoding_nerf_1d(coords: torch.Tensor, length: int):
    """
        Args:
            coords: b x 1
            length: total length
    """
    L = length // 2
    multiplier = (2 ** torch.arange(L)).float().to(coords.device).reshape(1, L) * torch.pi
    sin_part = torch.sin(multiplier * coords)
    cos_part = torch.cos(multiplier * coords)
    return torch.cat((sin_part, cos_part), dim=1)

def itertools_combinations(x: np.ndarray, r: int):
   
    idx = np.stack(list(itertools.combinations(x, r=r)))

    return x[idx]
