import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
from training.networks_stylegan2 import FullyConnectedLayer
from hash_encoding.other_networks import MappingNetwork
from hash_encoding.modules import StackedModulatedMLP
from utils.utils import pos_encoding_nerf_1d

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution,
                calc_grad_inputs=False,
                gridtype=0,
                align_corners=False,
                res_multiplier=1,
                feat_coord_dim=1):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=embeddings.dtype) # placeholder... TODO: a better way?

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, calc_grad_inputs,
                                     dy_dx, gridtype, align_corners, res_multiplier, feat_coord_dim)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.align_corners = align_corners
        ctx.res_multiplier = res_multiplier
        ctx.feat_coord_dim = feat_coord_dim

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        align_corners = ctx.align_corners
        res_multiplier = ctx.res_multiplier
        feat_coord_dim = ctx.feat_coord_dim

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=embeddings.dtype)

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs,
                                      dy_dx, grad_inputs, gridtype, align_corners, res_multiplier, feat_coord_dim)

        if calc_grad_inputs:
            grad_inputs = grad_inputs.to(inputs.dtype)
            return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class GridEncoder(nn.Module):
    def __init__(self,
                 input_dim=3,
                 num_levels=16,
                 level_dim=2,
                 per_level_scale=2,
                 base_resolution=4,
                 log2_hashmap_size=19,
                 desired_resolution=64,
                 gridtype='hash',
                 align_corners=False,
                 style_dim=256,
                 feat_coord_dim=2,
                 out_dim=16,
                 out_dim2=16,
                 dummy_hash_table=False,
                 tile_coord=True,
                 mini_linear_n_layers=3,
                 no_modulated_linear=False,
                 direct_coord_input=False,
                 one_hash_group=False,
                 fused_spatial=False,
                 res_multiplier=1,
                 pg_hash_res=False,
                 pg_hr_iter_k=20,
                 pg_init_method='replicate',
                 pg_detach=False,
                 pg_alter_opti=False,
                 pg_init_iter=0
                 ):
        """
            Args:
                input_dim (int): Total retrive coordinates to the hash table.
                    (default: 3)
                num_levels (int): The resolution levels in hash tables. 
                    (default: 16)
                level_dim (int): The dimension of each entries in hash table.
                    (default: 2)
                per_level_scale (float): The scaling of resolutions among 
                    hash tables. If desired resolution (max resolution)
                    is provided, this value will be caculated based on 
                    desired resoluton instead. (default: 2)
                base_resolution (int): Basic resolution (smallest res).
                    (default: 4)
                log2_hashmap_size (int): Log2 hashmap size for each hash table.
                    (default: 19)
                desired_resolution (int): Desired resolution (max res).
                    (default: 64)
                gridtype (str): Grid type. (default: 'hash')
                align_corners (bool): Align corners. (default: False)
                style_dim (int): style code input dimension to do modulation or
                    mapping to coord. (default: 256)
                feat_coord_dim (int): The mapped dimension of the feature 
                    coordinate. (default: 2)
                out_dim (int): Final output feature of this module (after 
                    pointwise MLP mapping). (default: 16)
                dummy_hash_table (bool): Dummy hash table. Instead of retrieve 
                    it is the constant. (default: False)
                tile_coord (bool): Tiling the feature coord with spatial 
                    coordinates. (default: True)
                mini_linear_n_layers (int): The MLP mapping number of layers
                    after retrieving. (default: 3)
                no_modulated_linear (bool): If true, the MLP will not be modulated.
                    (default: False)
                direct_coord_input (bool): The feature coord is by input instead
                    of internal mapping. (default: False)
                one_hash_group (bool): Only one hash group. If so, no modulation,
                    output actual style code at the same time.
                fused_spatial (bool): If fused spatial, we would use the additional
                    input coordinates to modulate that.
                pg_hash_res: Progressive hash resolution ratio flag
                    (default: False)
                pg_hr_iter_k: Progressive hash resolution iteration k
                    (default: 20)
                pg_init_method: Progressive hash resolution initialization method
                    (default: replicate)
                pg_detach: Detach encoded key codes when increase the resolution
                    (default: False)
                pg_alter_opti: Progressive increase resolution and alternatively
                    optimize indices and feature grids.
                    (default: False)
                pg_init_iter: The initial joint training step of progressive training.
                    (default: 0)
        """
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.hash_out_dim = num_levels * level_dim
        self.out_dim = out_dim
        self.out_dim2 = out_dim2
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners
        self.feat_coord_dim = feat_coord_dim
        self.dummy_hash_table = dummy_hash_table
        self.tile_coord = tile_coord
        self.out_total_pixel = desired_resolution * desired_resolution
        self.one_hash_group = one_hash_group
        self.no_modulated_linear = no_modulated_linear
        self.direct_coord_input = direct_coord_input
        self.fused_spatial = fused_spatial
        self.res_multiplier = res_multiplier
        self.pg_hash_res = pg_hash_res
        self.pg_hr_iter_k = pg_hr_iter_k
        self.pg_init_method = pg_init_method
        self.pg_detach = pg_detach
        self.pg_alter_opti = pg_alter_opti
        self.pg_init_iter = pg_init_iter

        if self.pg_hash_res:
            self.register_buffer('pg_iter_count', torch.zeros(1) + 1)


        if one_hash_group:
            style_dim = self.hash_out_dim * 2


        if self.dummy_hash_table:
            self.register_buffer('const_input', torch.randn(1,
                                                            desired_resolution*desired_resolution,
                                                            self.hash_out_dim,
                                                            ))

        if not direct_coord_input:
            s_out_dim = feat_coord_dim if tile_coord else feat_coord_dim * desired_resolution * desired_resolution
            self.style_mapping = FullyConnectedLayer(style_dim, s_out_dim,
                                                    activation='sigmoid',
                                                    bias_init=1)

        if fused_spatial:
            self.mini_linear4 = MappingNetwork(in_ch=32,
                                                map_depth=2,
                                                split_depth=1,
                                                hidden_ch=self.hash_out_dim,
                                                style_dim=self.hash_out_dim,
                                                use_layer_norm=False,
                                                two_style_code=False
                                                )

        if not no_modulated_linear:
            self.mini_linear = StackedModulatedMLP(self.hash_out_dim,
                                                self.hash_out_dim,
                                                out_dim,
                                                style_dim,
                                                mini_linear_n_layers,
                                                in_activ=nn.LeakyReLU,
                                                out_activ=nn.Identity
                                                )
        else:
            self.mini_linear = MappingNetwork(in_ch=self.hash_out_dim,
                                              map_depth=mini_linear_n_layers,
                                              style_dim=out_dim,
                                              hidden_ch=self.hash_out_dim,
                                              use_layer_norm=False,
                                              two_style_code=False)

        if one_hash_group:
            self.mini_linear2 = MappingNetwork(in_ch=self.hash_out_dim,
                                               map_depth=mini_linear_n_layers,
                                               style_dim=out_dim2,
                                               hidden_ch=self.hash_out_dim,
                                               lr_multiplier=0.01,
                                               use_layer_norm=False,
                                               two_style_code=False)
                                               
            self.mini_linear3 = MappingNetwork(in_ch=self.hash_out_dim,
                                               map_depth=2,
                                               style_dim=style_dim,
                                               hidden_ch=self.hash_out_dim,
                                               lr_multiplier=0.01,
                                               use_layer_norm=False,
                                               two_style_code=False)

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            resolution = resolution if align_corners else resolution + 1
            params_in_level_ = resolution ** input_dim
            if res_multiplier > 1:
                params_in_level_ = params_in_level_ * int(res_multiplier ** feat_coord_dim)
            params_in_level = min(self.max_params, params_in_level_) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, s=None, modulation_s=None, bound=1, b=None, mod_coords=None):
        """
            Args:
                inputs: [..., input_dim], normalized real world positions in [-bound, bound]
                s: [..., style_dim],  the style code, to encode the style coordinates
                modulation_s: to modulate the minilinear if provided.
                return: [..., num_levels * level_dim]
        """
        if self.one_hash_group and self.direct_coord_input:
            inputs = inputs
        elif not self.no_modulated_linear:
            b = s.shape[0]
            if self.dummy_hash_table:
                outputs = self.const_input.repeat((b, 1, 1))
                outputs = self.mini_linear(outputs, s)
                outputs = outputs.reshape(-1, self.out_dim)
                return outputs

            s_coords = self.style_mapping(s) if not self.direct_coord_input else s
            if self.tile_coord:
                s_coords = s_coords.unsqueeze(dim=1).repeat((1, self.out_total_pixel, 1))
            s_coords = s_coords.reshape(-1, self.feat_coord_dim)
            inputs = torch.cat([inputs, s_coords], dim=-1)
        else:
            b = inputs.shape[0]
            inputs = self.style_mapping(inputs) if not self.direct_coord_input else inputs
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        # Progressive enlarge the table or not?
        res_multiplier_ori = res_multiplier = getattr(self, "res_multiplier", 1)
        feat_coord_dim = getattr(self, "feat_coord_dim", 1)
        offsets_ori = offsets = self.offsets
        embeddings = self.embeddings
        if getattr(self, "pg_hash_res", False):
            # The facotr `k`
            init_iter_n = getattr(self, "pg_init_iter", 0)
            count = self.pg_iter_count.cpu().item() - init_iter_n
            k = max(count // (self.pg_hr_iter_k * 1000), 0)
            if (k > 0 or (k >= 0 and init_iter_n > 0)) and self.pg_detach:
                inputs = inputs.detach()
            elif getattr(self, "pg_alter_opti", False):
                k_ = count // (self.pg_hr_iter_k * 500)
                k_ = k_ - 2
                if k_ >= 0:
                    if k_ % 2 == 0:
                        inputs = inputs.detach()
                    if k_ % 2 == 1:
                        embeddings = self.embeddings.detach()
            # The divide ratio
            l = int(max(int(res_multiplier_ori/(int(2**k))), 1))
            # Decrease the offsets if necessary
            offsets = (offsets_ori / l).to(torch.int32)
            # Gradually increase res_multiplier
            res_multiplier = min(int(2 ** k), res_multiplier_ori)
            if (count > 0) and (count % (self.pg_hr_iter_k * 1000) == 0):
                if int(2 ** k) <= res_multiplier:
                    print("Increase the embeddings now! During the progressive growing!")
                    print(f"multiplier now {2 ** k} vs {res_multiplier_ori}")
                    l = int(max(int(res_multiplier_ori/(int(2**k))), 1))
                    if self.pg_init_method == "replicate":
                        # Initialize the adjacent replicate to ensure smooth transition
                        embeddings[l::int(l*2), :].data = embeddings[::int(l*2), :].data
                    elif self.pg_init_method == "median":
                        # Initialize the adjacent via median values
                        e_ = embeddings[::int(l*2), :].data
                        embeddings[l::int(l*2), :][:-1, :].data = (e_[1:, :] + e_[:-1, :]) / 2.0
                    elif self.pg_init_method == "none":
                        pass
                    # Recalculate the offsets
                    offsets = (offsets_ori / l).to(torch.int32)
            embeddings = embeddings[::l, :]


        outputs = grid_encode(inputs,
                              embeddings,
                              offsets,
                              self.per_level_scale,
                              self.base_resolution,
                              inputs.requires_grad,
                              self.gridtype_id,
                              self.align_corners,
                              res_multiplier,
                              feat_coord_dim
                              )
        outputs = outputs.view(prefix_shape + [self.hash_out_dim])

        # This original outputs is the hash retrieve outpout before 
        outputs = outputs.reshape(b, -1, self.hash_out_dim)


        # Second output (output style code at the time of outputing spatical feats)
        if self.one_hash_group:
            outputs2 = normalize_2nd_moment(outputs.mean(dim=1))
            modulation_s, _ = self.mini_linear3(outputs2)
            outputs2, _ = self.mini_linear2(normalize_2nd_moment(outputs2))
            outputs2 = outputs2.reshape(-1, self.out_dim2)
        else:
            outputs2 = None
            
        if not self.no_modulated_linear:
            if getattr(self, 'fused_spatial', False):
                beta_f, _ = self.mini_linear4(pos_encoding_nerf_1d(mod_coords, 32))
                beta_f = beta_f.reshape(outputs.shape)
                outputs = outputs  + beta_f

            outputs1 = self.mini_linear(outputs, modulation_s)
        else:
            outputs1 = outputs.reshape(-1, self.hash_out_dim)
            outputs1, _ = self.mini_linear(outputs1)

        outputs1 = outputs1.reshape(-1, self.out_dim)

        # Add the iters
        if getattr(self, 'pg_hash_res', False) and self.training:
            self.pg_iter_count += 1

        return outputs1, outputs2


class VarGridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, hash_entries=None):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.hash_out_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim
        self.level_dim = level_dim
        self.offset = offset

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset - hash_entries, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, embeddings, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        input_embeddings = torch.cat([embeddings, self.embeddings], dim=0)

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, input_embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners)
        outputs = outputs.view(prefix_shape + [self.hash_out_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs