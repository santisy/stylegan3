#!/usr/bin/env python3
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import namedtuple
from functools import partial
from glob import glob
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
import time

import numpy as np
import cupy as cp
import PIL
from PIL import Image

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return x

    def trange(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return range(x)

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


from utils.simple_dataset import SimpleDatasetForMetric

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])

PR_PROCESS_NUM = int(os.getenv('PR_PROCESS_NUM', 4))


class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=10000, model=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        if model is None:
            print('loading vgg16 for improved precision and recall...', end='', flush=True)
            self.vgg16 = models.vgg16(pretrained=True).cuda().eval()
            print('done')
        else:
            self.vgg16 = model

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
        recall = compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, path, save_path=None):
        self.manifold_ref = self.compute_manifold(path, save_path)

    def realism(self, image):
        '''
        args:
            image: torch.Tensor of 1 x C x H x W
        '''
        feat = self.extract_features(image)
        return realism(self.manifold_ref, feat)

    def compute_manifold(self, input, save_path=None):
        '''
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        '''
        if save_path is not None and os.path.isfile(save_path):
            input = save_path

        # features
        if isinstance(input, str):
            if input.endswith('.npz'):  # input is precalculated file
                print('loading', input)
                f = np.load(input)
                feats = f['feature']
                radii = f['radii']
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self.extract_features_from_files(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extract_features(input)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extract_features(input)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extract_features_from_files(input)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError

        compute_radii_from_feats = ComputeRadiiFromFeats(feats, k=self.k)
        start_time = time.time()
        pool = Pool(processes=PR_PROCESS_NUM)
        pool.map(compute_radii_from_feats, tuple(range(feats.shape[0])))
        pool.close()
        pool.join()
        radii = np.asarray(list(compute_radii_from_feats.radii)).flatten()
        end_time = time.time()
        print(f'\033[93mTotal calculating radius time:{end_time - start_time}s\033[00m')
        return Manifold(feats, radii)

    def extract_features(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            print('IPR: resizing %s to (224, 224)' % str((height, width)))
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x): return x

        features = []
        for bi in trange(num_batches, desc=desc):
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = images[start:end]
            batch = resize(batch)
            before_fc = self.vgg16.features(batch.cuda())
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def extract_features_from_files(self, path_or_fnames):
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """

        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)
        num_found_images = len(dataloader.dataset)
        desc = 'extracting features of %d images' % num_found_images
        if num_found_images < self.num_samples:
            print('WARNING: num_found_images(%d) < num_samples(%d)' % (num_found_images, self.num_samples))

        features = []
        for batch in tqdm(dataloader, desc=desc):
            before_fc = self.vgg16.features(batch.cuda())
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def save_ref(self, fname):
        print('saving manifold to', fname, '...')
        np.savez_compressed(fname,
                            feature=self.manifold_ref.features,
                            radii=self.manifold_ref.radii)


def compute_pairwise_distances(X, Y=None):
    '''
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    '''
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
    else:
        num_Y = Y.shape[0]
    X = X.astype(np.float64)  # to prevent underflow
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)
    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    diff_square = X_square - 2*XY + Y_square

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
              min_diff_square)

    distances = np.sqrt(diff_square)
    return distances

class ComputeMetric(object):
    def __init__(self, mani_ref, feats):
        self.mani_ref = mani_ref
        self.ref_radii = self._to_cuda(mani_ref.radii)
        self.feats_ref = self._to_cuda(mani_ref.features)
        self.feats_ref_square = cp.sum(self.feats_ref ** 2, axis=1, keepdims=True)
        self.feats = self._to_cuda(feats)
        self.feats_square = cp.sum(self.feats ** 2, axis=1, keepdims=True)
        self._count_list = Manager().list([0] * feats.shape[0])

    @property
    def result(self):
        return float(sum(list(self._count_list)))

    def _to_cuda(self, x):
        return cp.asarray(x)

    def __call__(self, i):
        feat_now = self.feats[i][None, :]
        dot_ = cp.sum(feat_now * self.feats_ref, axis=1, keepdims=True)
        distance = cp.sqrt(self.feats_square[i][None, :] - 2 * dot_ + self.feats_ref_square)
        distance = distance.flatten()
        distance[distance < 0] = 0
        self._count_list[i] += int((distance < self.ref_radii).any().get())


class ComputeRadiiFromFeats(object):
    def __init__(self, feats, k):
        feats = self._to_cuda(feats)
        self.feats = feats
        self.feats_square = cp.sum(feats ** 2, axis=1, keepdims=True)
        self.k = k
        self._radii = Manager().list([0] * feats.shape[0])
    
    @property
    def radii(self):
        return self._radii

    def _to_cuda(self, x):
        return cp.asarray(x)

    def __call__(self, i):
        feat_now = self.feats[i][None, :]
        dot_ = cp.sum(feat_now * self.feats, axis=1, keepdims=True)
        distance = cp.sqrt(self.feats_square[i][None, :] - 2 * dot_ + self.feats_square)
        distance = distance.flatten()
        distance[distance < 0] = 0
        self._radii[i] = get_kth_value(distance, k=self.k)

def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = cp.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max().get()
    return kth_value


def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    cm = ComputeMetric(manifold_ref, feats_subject)
    pool = Pool(processes=PR_PROCESS_NUM // 2 + 1)
    pool.map(cm, tuple(range(feats_subject.shape[0])))
    pool.close()
    pool.join()
    return cm.result / num_subjects


def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def realism(manifold_real, feat_subject):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    eps = 1e-6
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism

class ZipDataset(SimpleDatasetForMetric):
    def __init__(self, root, transform=None, num_samples=-1):
        super().__init__(root, torch.device('cpu'), max_num=num_samples)
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        print(f'\033[92mIterating path {root} with {self.data_len} \033[00m')
        self.transform = transform

    def __getitem__(self, index):
        fname = self._image_fnames[index]
        with self._open_file(fname) as f:
            image = PIL.Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.png'), recursive=True)
        print(f'\033[92mIterating path {root} with {len(self.fnames)} \033[00m')
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


class FileNames(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


def get_custom_loader(image_dir_or_fnames, image_size=224, batch_size=50, num_workers=4, num_samples=-1):
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)

    if isinstance(image_dir_or_fnames, list):
        dataset = FileNames(image_dir_or_fnames, transform)
    elif isinstance(image_dir_or_fnames, str):
        if image_dir_or_fnames.endswith(".zip"):
            dataset = ZipDataset(image_dir_or_fnames,
                                 transform=transform,
                                 num_samples=num_samples)
        else:
            dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError

    if num_samples > 0 and not image_dir_or_fnames.endswith(".zip"):
        dataset.fnames = dataset.fnames[:num_samples]
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=False,
                             multiprocessing_context="fork")
    return data_loader


def toy():
    offset = 2
    feats_real = np.random.rand(10).reshape(-1, 1)
    feats_fake = np.random.rand(10).reshape(-1, 1) + offset
    feats_real[0] = offset
    feats_fake[0] = 1
    print('real:', feats_real)
    print('fake:', feats_fake)

    print('computing pairwise distances...')
    distances_real = compute_pairwise_distances(feats_real)
    print('distances to radii...')
    radii_real = distances2radii(distances_real)
    manifold_real = Manifold(feats_real, radii_real)

    print('computing pairwise distances...')
    distances_fake = compute_pairwise_distances(feats_fake)
    print('distances to radii...')
    radii_fake = distances2radii(distances_fake)
    manifold_fake = Manifold(feats_fake, radii_fake)

    precision = compute_metric(manifold_real, feats_fake)
    recall = compute_metric(manifold_fake, feats_real)
    print('precision:', precision)
    print('recall:', recall)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print(f'\033[93mThe process num is {PR_PROCESS_NUM}\033[00m.')

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_real', type=str, help='Path to the real images')
    parser.add_argument('path_fake', type=str, help='Path to the fake images')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size to use')
    parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fname_precalc', type=str, default='', help='fname for precalculating manifold')
    parser.add_argument('--only_precalc', action='store_true', default=False)
    args = parser.parse_args()

    # toy problem
    if args.toy:
        print('running toy example...')
        toy()
        exit()

    # Example usage: with real and fake paths
    # python improved_precision_recall.py [path_real] [path_fake]
    ipr = IPR(args.batch_size, args.k, args.num_samples)
    with torch.no_grad():
        real_name = os.path.basename(args.path_real)
        save_path = os.path.join('metrics_cache', f'{real_name}_PR_stats')

        # real
        ipr.compute_manifold_ref(args.path_real, save_path=save_path + '.npz')

        # save and exit for precalc
        if not os.path.isfile(save_path + '.npz'):
            ipr.save_ref(save_path)
            if args.only_precalc:
                exit()

        # fake
        precision, recall = ipr.precision_and_recall(args.path_fake)

    print('precision:', precision)
    print('recall:', recall)

    f = open(os.path.join("metrics_cache",
                          f"{os.path.basename(args.path_fake)}_precision_recall.txt"),
                          'a')
    f.write(f"real data: {args.path_real}\n")
    f.write(f"fake data: {args.path_fake}\n")
    f.write(f"precision: {precision}")
    f.write(f"recall: {recall}")
