"""Some utilities"""

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
