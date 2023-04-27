"""Validate the extracted nc is Okay."""
import cv2
import numpy as np
import torch

import dnnlib
import legacy
from diffusions.decode import decode_nc


with dnnlib.util.open_url('training_runs/en_0422_01/network-snapshot-001000.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    G = G.eval()
    G = G.cuda()

with open('exported_nc/en_0422_01_LSUN_Church_256x256_val/0000102.npy', 'rb') as f:
    out = np.load(f)
    nc = torch.from_numpy(out['arr_0'].copy()).cuda()


def decode_and_write_image(img_name: str, nc: torch.Tensor):
    nc = torch.clip(nc, 0, 1)
    with torch.no_grad():
        out = decode_nc(G, nc)
    img = (out[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{img_name}.png', img)

#sigmas = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
#
#for sigma in sigmas:
decode_and_write_image(f'test', nc)
