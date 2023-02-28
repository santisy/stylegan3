"""Some utilities"""
import os

import torch

__all__ = ['sinuous_pos_encode']


def sinuous_pos_encode(table_num: int, token_n: int):
    """
        Args:
            table_num (int): table number for specify the index for each table
            token_n (int): frequency encoded dimension. The length of each input
                token.
        Returns:
            positional encoded vector B x N x (dim x 2)
    """

    indices = torch.arange(table_num).unsqueeze(dim=1).float()
    k = (torch.arange(token_n).float() / token_n).unsqueeze(dim=0)

    temp_p = indices ** k
    temp_p = temp_p.unsqueeze(dim=0) # 1 x N (number of token) x L (token length)
    
    # Phase shift to generate both sin and cos
    shift_p = torch.zeros((token_n // 2, 2))
    shift_p[:, 1] = torch.pi / 2.
    shift_p = shift_p.flatten()
    shift_p = shift_p.unsqueeze(dim=0).unsqueeze(dim=0)

    return torch.cos(temp_p + shift_p)


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
