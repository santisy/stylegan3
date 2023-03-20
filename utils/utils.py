"""Some utilities"""
import os

import numpy as np
import torch
import torch.nn.functional as F

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
    shift_p[:, 1] = torch.pi / 2.
    shift_p = shift_p.flatten()
    shift_p = shift_p.unsqueeze(dim=0).unsqueeze(dim=0)
    pos_encodings = torch.cos(temp_p + shift_p)

    for i, res in enumerate(res_list):
        dropout_p = np.clip(1 - (res / float(res_max)) ** 2, 0, 1)
        pos_encodings[0, i, :] = F.dropout(pos_encodings[0, i, :],
                                           p=dropout_p)

    pos_encodings = pos_encodings.reshape(1, table_num, token_d).detach()

    return pos_encodings


def delete_file(file_path: str):
    # TODO: More to add if we involve HDFS file system
    # Remove the local path
    os.remove(file_path)

def sample_coords(b: int, img_size: int,
                  sample_size: int=None,
                  single_batch: bool=False):
    """
        Args:
            b (int): batch_size
            img_size (int): image size
        Retrun:
            coords (torch.Tensor): B x N x (2 or 3), value range [0, 1]
    """
    # 2D sampling case
    c = torch.arange(img_size) + 0.5
    x, y = torch.meshgrid(c, c, indexing='xy')
    # Normalize it to [0, 1]
    coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2) / img_size

    if sample_size is not None:
        sampled_indices = torch.randperm(coords.shape[1])[:sample_size]
        coords = coords[:, sampled_indices, :]

    if not single_batch:
        coords = coords.repeat(b, 1, 1)

    return coords

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
