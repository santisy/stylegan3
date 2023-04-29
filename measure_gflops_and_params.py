"""Count the model GFlops and parameters
"""
import click

import legacy
import dnnlib


def conv_flops(res, in_ch, out_ch, style_dim=512, up=False):
    """StyleGAN convolution GFlops counting"""
    flops = 0

    if up:
        pre_res = res // 2
    else:
        pre_res = res

    # Style MLP
    flops += (style_dim * in_ch + in_ch)

    # Conv
    flops += (3 * 3 * pre_res * pre_res * in_ch * out_ch)

    # Modulation
    # 1) Do w * styles
    flops += (3 * 3 * in_ch * out_ch)
    # 2) Get decoefs
    flops += (3 * 3 * in_ch * out_ch * 2)
    # 3) Do w * decoefs
    flops += (3 * 3 * in_ch * out_ch)
    # 4) Demodulation
    flops += (res * res * out_ch)


    # Act gain
    flops += (res * res * out_ch)

    # Upfird
    if up:
        flops += (4 * 4 * res * res * out_ch) # Layer-wise operation

    return flops

def torgb_flops(res, in_ch, style_dim=512):
    """To RGB Layers"""
    flops = 0

    # Style MLP
    flops += (style_dim * in_ch + in_ch)

    # Conv
    flops += (1 * 1 * in_ch * 3)

    # Modulation
    # 1) Do w * styles
    flops += (1 * 1 * in_ch * 3)
    # 2) Get decoefs
    flops += (1 * 1 * in_ch * 3 * 2)
    # 3) Do w * decoefs
    flops += (1 * 1 * in_ch * 3)
    # 4) Demodulation
    flops += (res * res * 3)

    # Act gain
    flops += (res * res * 3)

    return flops

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--level_dim', type=int, default=4)
@click.option('--out_res', type=int, default=256)
def calc_flops_and_params(network_pkl: str,
                          level_dim: int=4,
                          out_res: int=256
                          ):

    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        G = G.eval()
    
    # Caculate Parameters
    hash_parameter_count = 0
    for hash_encoder in G.hash_encoder_list:
        hash_parameter_count += count_parameters(hash_encoder)
    decoder_parameter_count = count_parameters(G.synthesis_network)

    print('\033[93mHash encoder total parameter counts:'
          f' {hash_parameter_count/1e6:.4f}M\033[00m')
    print('\033[93mDecoder total parameter counts:'
          f' {decoder_parameter_count/1e6:.4f}M\033[00m')
    print('\033[93mTotal parameter memory:'
          f' {(decoder_parameter_count+hash_parameter_count)*4/1e9:.4f}Gb\033[00m')

    # Extract parameters
    init_res = G.init_res
    init_dim = G.init_dim
    style_dim = 512 # Default to 512, fixed
    feat_coord_dim = G.feat_coord_dim

    flops = 0
    res_now = init_res
    dim_now = init_dim

    # Rtrieve from hash encoder
    # 1) Trieve and interpolate
    flops += (init_res * init_res * feat_coord_dim * (8 * 2) * level_dim) 
    # 2) MLPs
    flops += (init_res * init_res * ((level_dim * 16) ** 2 * 2 + level_dim * 16 * init_dim // feat_coord_dim) * feat_coord_dim)

    # StyleGAN synthesis layer flops
    while res_now <= out_res:
        flops += conv_flops(res_now, dim_now, dim_now, style_dim, up=False) * 2
        if res_now != init_res:
            flops += conv_flops(res_now, dim_now, dim_now, style_dim, up=True)
        flops += torgb_flops(res_now, dim_now, style_dim)

        # Change to the next layer
        res_now = res_now * 2
        dim_now = dim_now // 2

    gflops = flops / 1e9
    print(f'\033[93mGFlops is {gflops:.2f}\033[00m')


if __name__ == '__main__':
    calc_flops_and_params()