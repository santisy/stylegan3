"""Simple dataset for iterating the images
"""
import os
import zipfile

import PIL.Image
import numpy as np
import torch

__all__ = ['SimpleDataset', 'SimpleDatasetForMetric']

class SimpleDataset:
    def __init__(self,
                 dataset: str,
                 device: torch.device,
                 last_split_n: int=-1):
        """
            Simple dataset to extract the images from zip file and return a 
            batch size 1 CUDA tensor.
            Args:
                dataset (str): where the zip dataset files locate.
                device (torch.device): what is the torch device
                last_split_n (int): If bigger than 0, only iterate last n
                    images. (default: -1)
        """

        assert os.path.isfile(dataset) and dataset.endswith('zip')
        self._data_path = dataset
        self._zipfile = None
        self._all_fnames = set(self._get_zipfile().namelist())
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames
                                    if self._file_ext(fname)
                                    in PIL.Image.EXTENSION)
        if last_split_n > 0:
            self._image_fnames = self._image_fnames[:-last_split_n]
        self.data_len = len(self._image_fnames)
        self.device = device

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        return self._get_zipfile().open(fname, 'r')

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._data_path)
        return self._zipfile
    
    def retrieve_by_idx(self, idx) -> PIL.Image.Image:
        with self._open_file(self._image_fnames[idx]) as f:
            image = np.array(PIL.Image.open(f)).copy()
        return image

    def __len__(self):
        return self.data_len

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.data_len:
            fname = self._image_fnames[self.n]
            self.n += 1
            with self._open_file(fname) as f:
                image = np.array(PIL.Image.open(f))
            image = image.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(image.copy()).float()
            img_tensor = img_tensor / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(dim=0).to(self.device)
            return img_tensor
        else:
            raise StopIteration

class SimpleDatasetForMetric(torch.utils.data.Dataset):
    def __init__(self,
                 dataset: str,
                 device: torch.device):
        """
            Simple dataset to extract the images from zip file and return a 
            batch size 1 CUDA tensor (uint8 as required for FID calculation).
            Args:
                dataset (str): where the zip dataset files locate.
                device (torch.device): what is the torch device
        """

        assert os.path.isfile(dataset) and dataset.endswith('zip')
        self._data_path = dataset
        self._zipfile = None
        self._all_fnames = set(self._get_zipfile().namelist())
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames
                                    if self._file_ext(fname)
                                    in PIL.Image.EXTENSION)
        self.data_len = len(self._image_fnames)
        self.device = device

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._data_path)
        return self._zipfile

    def _open_file(self, fname):
        return self._get_zipfile().open(fname, 'r')

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        fname = self._image_fnames[idx]
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f))
        image = image.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(image.copy())
        return img_tensor
