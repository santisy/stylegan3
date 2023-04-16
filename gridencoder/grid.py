import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 
from training.networks_stylegan2 import FullyConnectedLayer
from hash_encoding.other_networks import MappingNetwork
from hash_encoding.modules import StackedModulatedMLP

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
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

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

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, calc_grad_inputs, dy_dx, gridtype, align_corners)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.align_corners = align_corners

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=embeddings.dtype)

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs, gridtype, align_corners)

        if calc_grad_inputs:
            grad_inputs = grad_inputs.to(inputs.dtype)
            return grad_inputs, grad_embeddings, None, None, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self,
                 input_dim=3, num_levels=16, level_dim=2,
                 per_level_scale=2, base_resolution=16, log2_hashmap_size=19,
                 desired_resolution=None, gridtype='hash',
                 align_corners=False, style_dim=256, feat_coord_dim=16,
                 out_dim=64, dummy_hash_table=False, tile_coord=False,
                 mini_linear_n_layers=3, no_modulated_linear=False,
                 ):
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
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners
        self.feat_coord_dim = feat_coord_dim
        self.dummy_hash_table = dummy_hash_table
        self.tile_coord = tile_coord
        self.out_total_pixel = desired_resolution * desired_resolution
        self.no_modulated_linear = no_modulated_linear

        if self.dummy_hash_table:
            self.register_buffer('const_input', torch.randn(1,
                                                            desired_resolution*desired_resolution,
                                                            self.hash_out_dim,
                                                            ))

        s_out_dim = feat_coord_dim if tile_coord else feat_coord_dim * desired_resolution * desired_resolution
        self.style_mapping = FullyConnectedLayer(style_dim, s_out_dim,
                                                 activation='sigmoid',
                                                 bias_init=1)

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

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, s=None, modulation_s=None, bound=1):
        """
            Args:
                inputs: [..., input_dim], normalized real world positions in [-bound, bound]
                s: [..., style_dim],  the style code, to encode the style coordinates
                modulation_s: to modulate the minilinear if provided.
                return: [..., num_levels * level_dim]
        """
        if not self.no_modulated_linear:
            b = s.shape[0]
            if self.dummy_hash_table:
                outputs = self.const_input.repeat((b, 1, 1))
                outputs = self.mini_linear(outputs, s)
                outputs = outputs.reshape(-1, self.out_dim)
                return outputs

            s_coords = self.style_mapping(s)
            if self.tile_coord:
                s_coords = s_coords.unsqueeze(dim=1).repeat((1, self.out_total_pixel, 1))
            s_coords = s_coords.reshape(-1, self.feat_coord_dim)
            inputs = torch.cat([inputs, s_coords], dim=-1)
        else:
            b = inputs.shape[0]
            inputs = self.style_mapping(inputs)
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners)
        outputs = outputs.view(prefix_shape + [self.hash_out_dim])

        outputs = outputs.reshape(b, -1, self.hash_out_dim)
        if not self.no_modulated_linear:
            outputs = self.mini_linear(outputs, modulation_s)
        else:
            outputs = outputs.reshape(b, self.hash_out_dim)
            outputs, _ = self.mini_linear(outputs)
        outputs = outputs.reshape(-1, self.out_dim)

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

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