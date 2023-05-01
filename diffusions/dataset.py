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
                 size: int=16,
                 use_kl_reg: bool=False,
                 noise_perturb: bool=False,
                 noise_perturb_sigma: float=-1.0,
                 ):
        """
            Args:
                folder_path (str): The extracted neural coordinates 'npy' file
                    folder path.
                dim (int): neural coordinates dimension, the channel wise
                    (default: 16)
                size (int): The spatial size of the neural coordinates.
                    (default: 16)
                use_kl_reg (bool): If using KL regularization or not.
                    (default: False)
                noise_perturb (bool): If using noise perturbation.
                    (default: False)
                noise_perturb_sigma (float): noise_perturbation sigma. 
                    (default: -1.0)
        """
        self._npy_files = glob.glob(os.path.join(folder_path, f'*.npz'))

        self._dim = dim
        self._size = size
        self._use_kl_reg = use_kl_reg
        self._noise_perturb = noise_perturb
        self._noise_perturb_sigma = noise_perturb_sigma

    def __len__(self):
        return len(self._npy_files)


    def __getitem__(self, idx):
        npy_file = self._npy_files[idx]
        with open(npy_file, 'rb') as f:
            loaded = np.load(f)
            if not self._use_kl_reg:
                ni = loaded['ni'][0]
            else:
                mu = loaded['mu'][0]
                log_var = loaded['log_var'][0]
                ni = np.random.normal(size=mu.shape) * np.exp(0.5 * log_var) + mu

        if ni.shape[0] != self._dim:
            raise RuntimeError('Dimension of neural coordinates wrong.'
                               f' required {self._dim} vs {ni.shape[0]}')

        if ni.shape[1] != self._size:
            raise RuntimeError('Size of neural coordinates wrong.'
                               f' required {self._size} vs {ni.shape[1]}')

        if self._noise_perturb:
            ni = np.random.normal(size=ni.shape) * self._noise_perturb_sigma + ni

        return ni
