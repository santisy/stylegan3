"""
    Extracted Neural coordinates diffusion training pytorch dataset
"""
import os
import glob

import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder_path: str,
                 dim: int=16,
                 size: int=16):
        """
            Args:
                folder_path (str): The extracted neural coordinates 'npy' file
                    folder path.
                dim (int): neural coordinates dimension, the channel wise
                    (default: 16)
                size (int): The spatial size of the neural coordinates.
                    (default: 16)
        """
        self._npy_files = glob.glob(os.path.join(folder_path, f'*.npy'))

        self._dim = dim
        self._size = size

    def __len__(self):
        return len(self._npy_files)


    def __getitem__(self, idx):
        npy_file = self._npy_files[idx]
        with open(npy_file, 'rb') as f:
            nc_numpy = np.load(f)[0]

        if nc_numpy.shape[0] != self._dim:
            raise RuntimeError('Dimension of neural coordinates wrong.'
                               f' required {self._dim} vs {nc_numpy.shape[0]}')

        if nc_numpy.shape[1] != self._size:
            raise RuntimeError('Size of neural coordinates wrong.'
                               f' required {self._size} vs {nc_numpy.shape[1]}')

        return nc_numpy
