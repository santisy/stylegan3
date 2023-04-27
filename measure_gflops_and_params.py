import click

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


@click.command()
@click.option('--init_res', type=int, default=64)
@click.option('--init_dim', type=int, default=512)
@click.option('--out_res', type=int, default=256)
@click.option('--style_dim', type=int, default=512)
@click.option('--feat_coord_dim', type=int, default=32)
@click.option('--level_dim', type=int, default=4)
def calc_flops(init_res: int=64,
               init_dim: int=512,
               out_res: int=256,
               style_dim: int=512,
               feat_coord_dim: int=32,
               level_dim: int=4,
               ):
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
    calc_flops()
