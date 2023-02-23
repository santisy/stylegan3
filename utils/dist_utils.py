"""Distribution training utils"""

import torch
import torch.distributed as tdist

__all__ = ['dprint']

def dprint(print_str: str, color=''):
    if not tdist.is_initialized() or tdist.get_rank() == 0:
        if color == '':
            print(print_str)
        elif color == 'r': # Red
            print(f'\033[91m{print_str}\033[00m')
        elif color == 'g': # Green
            print(f'\033[92m{print_str}\033[00m')
        elif color == 'y': # Yellow
            print(f'\033[93m{print_str}\033[00m')
        elif color == 'b': # Blue
            print(f'\033[94m{print_str}\033[00m')
        else:
            raise ValueError(f'Undefined color printing {color}.')
