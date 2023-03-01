/*
    Retrieve and interp from hash table.
    Author: Dingdong Yang
    Contacts: dya62@sfu.ca
*/
#include <torch/extension.h>
#include <ATen/cuda/Atomic.cuh>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

namespace{


template <typename scalar_t>
__global__ void interp_3D_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> features, // B x N x 8 x F
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords, // B x N x 3
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interp_feats // B x N x F
){
    const int b = blockIdx.y; // Batch size
    const int n = blockIdx.x * blockDim.x + threadIdx.x; // For the element to interp
    if (n >= features.size(1)){
        return;
    }
    scalar_t wx = coords[b][n][0] - floor(coords[b][n][0]);
    scalar_t wy = coords[b][n][1] - floor(coords[b][n][1]);
    scalar_t wz = coords[b][n][2] - floor(coords[b][n][2]);
    scalar_t wxy = wx * wy;
    for (int i=0; i < features.size(3); i++){
        scalar_t r1 = features[b][n][0][i] * (1 - wxy) + features[b][n][1][i] * (wy - wxy) +
                features[b][n][2][i] * (wx - wxy) + features[b][n][3][i] * wxy;
        scalar_t r2 = features[b][n][4][i] * (1 - wxy) + features[b][n][5][i] * (wy - wxy) +
                features[b][n][6][i] * (wx - wxy) + features[b][n][7][i] * wxy;
        interp_feats[b][n][i] = (1 - wz) * r1 + wz * r2;
    }
}

template <typename scalar_t, unsigned int F>
__global__ void retrieve_from_hash_table_2D_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hash_tables, // B x H_N x H_SIZE x F
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords, // B x N x 2, value range (0, 1)
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interp_feats, // B x N x (F * H_N)
    torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long T = hash_tables.size(2); // The 'T' notation is in coincide with the original instant-ngp paper
    const long h_n = hash_tables.size(1);

    if (n >= coords.size(1)){
        return;
    }

    // Const variables
    const long offsets[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const long primes[2] = {1, 2654435761};

    // Get the eight value
    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;

        scalar_t four_feats[4][F];
        for (int i = 0; i < 4; i++){
            const long query_n = (((x_f + offsets[i][0]) * primes[0]) ^
                                 ((y_f + offsets[i][1]) * primes[1])) % T;
            indices[b][n][j][i] = query_n;
            for (int k = 0; k < F; k++){
                four_feats[i][k] = hash_tables[b][j][query_n][k];
            }
        }

        for (int k = 0; k < F; k++){
            const auto r1 = four_feats[0][k] * wxy_1 + four_feats[1][k] * wy_wxy +
                            four_feats[2][k] * wx_wxy + four_feats[3][k] * wxy;
            interp_feats[b][n][j * F + k] = r1;
        }
    } 
}

template <typename scalar_t, unsigned int F>
__global__ void reconstruct_hash_table_2D_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> feats, // B x N x (H_N x F)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords, // B x N x 2
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hash_tables, // B x H_N x H_SIZE x F
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights, // B x H_N x H_SIZE x F
    torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long T = hash_tables.size(2); // The 'T' notation is in coincide with the original instant-ngp paper
    const long h_n = hash_tables.size(1);

    if (n >= coords.size(1)){
        return;
    }

    // Const variables
    const long offsets[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const long primes[2] = {1, 2654435761};

    // Get the eight value
    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;

        for (int i = 0; i < 4; i++){
            const long query_n = (((x_f + offsets[i][0]) * primes[0]) ^
                                 ((y_f + offsets[i][1]) * primes[1])) % T;
            indices[b][n][j][i] = query_n;
            for (int k = 0; k < F; k++){
                scalar_t f_value = feats[b][n][j * F + k];
                scalar_t * addr_now = reinterpret_cast<scalar_t *>(&hash_tables[b][j][query_n][k]);
                scalar_t * addr_w_now = reinterpret_cast<scalar_t *>(&weights[b][j][query_n][k]);
                switch (i){
                    case 0:
                        gpuAtomicAdd(addr_now, f_value * wxy_1);
                        gpuAtomicAdd(addr_w_now, wxy_1);
                        break;
                    case 1:
                        gpuAtomicAdd(addr_now, f_value * wy_wxy);
                        gpuAtomicAdd(addr_w_now, wy_wxy);
                        break;
                    case 2:
                        gpuAtomicAdd(addr_now, f_value * wx_wxy);
                        gpuAtomicAdd(addr_w_now, wx_wxy);
                        break;
                    case 3:
                        gpuAtomicAdd(addr_now, f_value * wxy);
                        gpuAtomicAdd(addr_w_now, wxy);
                        break;
                }
            }
        }
    } 
}

template <typename scalar_t, unsigned int F>
__global__ void retrieve_from_hash_table_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hash_tables, // B x H_N x H_SIZE x F
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords, // B x N x 3, value range (0, 1)
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interp_feats, // B x N x (F * H_N)
    torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long T = hash_tables.size(2); // The 'T' notation is in coincide with the original instant-ngp paper
    const long h_n = hash_tables.size(1);

    if (n >= coords.size(1)){
        return;
    }

    // Const variables
    const long offsets[8][3] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0},
                         {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    const long primes[3] = {1, 2654435761, 805459861};

    // Get the eight value
    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t z = coords[b][n][2] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        long z_f = floor(z);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wz = z - z_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wz_1 = 1 - wz;
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;
        scalar_t eight_feats[8][F];
        for (int i = 0; i < 8; i++){
            const long query_n = (((x_f + offsets[i][0]) * primes[0]) ^
                                 ((y_f + offsets[i][1]) * primes[1]) ^
                                 ((z_f + offsets[i][2]) * primes[2])) % T;
            indices[b][n][j][i] = query_n;
            for (int k = 0; k < F; k++){
                eight_feats[i][k] = hash_tables[b][j][query_n][k];
            }
        }

        for (int k = 0; k < F; k++){
            const auto r1 = eight_feats[0][k] * wxy_1 + eight_feats[1][k] * wy_wxy +
                          eight_feats[2][k] * wx_wxy + eight_feats[3][k] * wxy;
            const auto r2 = eight_feats[4][k] * wxy_1 + eight_feats[5][k] * wy_wxy +
                          eight_feats[6][k] * wx_wxy + eight_feats[7][k] * wxy;
            interp_feats[b][n][j * F + k] = wz_1 * r1 + wz * r2;
        }
    } 
}


template <typename scalar_t, unsigned int F>
__global__ void retrieve_from_hash_table_2D_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> feature_grad, // B x N x (H_N x F)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords,  // B x N x 2
    const torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices, // B x N x H_N x 4
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hash_grad // B x H_N x H_S x F
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long h_n = indices.size(2);

    if (n >= coords.size(1)){
        return;
    }

    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;
        for (int i = 0; i < 4; i++){
            long index_now = indices[b][n][j][i];
            for (int k = 0; k < F; k++){
                scalar_t grad_now = feature_grad[b][n][j * F + k];
                scalar_t * addr_now = reinterpret_cast<scalar_t *>(&hash_grad[b][j][index_now][k]);
                switch (i){
                    case 0:
                        gpuAtomicAdd(addr_now, grad_now * wxy_1);
                        break;
                    case 1:
                        gpuAtomicAdd(addr_now, grad_now * wy_wxy);
                        break;
                    case 2:
                        gpuAtomicAdd(addr_now, grad_now * wx_wxy);
                        break;
                    case 3:
                        gpuAtomicAdd(addr_now, grad_now * wxy);
                        break;
                }
            }
        }
    }
}


template <typename scalar_t, unsigned int F>
__global__ void reconstruct_hash_table_2D_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> table_grad, // B x H_N x H_S x F
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords, // B x N x 2
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights, // B x H_N x H_S x F
    const torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices, // B x N x H_N x 4
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> feat_grad // B x N x (H_N x 2)
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long h_n = indices.size(2);

    if (n >= coords.size(1)){
        return;
    }

    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;

        for (int i = 0; i < 4; i++){
            long index_now = indices[b][n][j][i];
            for (int k = 0; k < F; k++){
                scalar_t grad_now = table_grad[b][j][index_now][k];
                scalar_t w_ = weights[b][j][index_now][k] + 1e-7;
                scalar_t * addr_now = reinterpret_cast<scalar_t *>(&feat_grad[b][n][j * F + k]);
                switch (i){
                    case 0:
                        gpuAtomicAdd(addr_now, grad_now * wxy_1 / w_);
                        break;
                    case 1:
                        gpuAtomicAdd(addr_now, grad_now * wy_wxy / w_);
                        break;
                    case 2:
                        gpuAtomicAdd(addr_now, grad_now * wx_wxy / w_);
                        break;
                    case 3:
                        gpuAtomicAdd(addr_now, grad_now * wxy / w_);
                        break;
                }
            }
        }
    }
}


template <typename scalar_t, unsigned int F>
__global__ void retrieve_from_hash_table_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> feature_grad, // B x N x (H_N x F)
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coords,  // B x N x 3
    const torch::PackedTensorAccessor32<long, 4, torch::RestrictPtrTraits> indices, // B x N x H_N x 8
    const float * resolution_list, // Resolution list
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hash_grad // B x H_N x H_S x F
){
    // Thread allocation
    const long b = blockIdx.y; // Batch size 
    const long n = blockIdx.x * blockDim.x + threadIdx.x; // Each coordinate element to interp

    // Size variables
    const long h_n = indices.size(2);

    if (n >= coords.size(1)){
        return;
    }

    for (int j = 0; j < h_n; j++){
        const scalar_t x = coords[b][n][0] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t y = coords[b][n][1] * static_cast<scalar_t>(resolution_list[j]);
        const scalar_t z = coords[b][n][2] * static_cast<scalar_t>(resolution_list[j]);
        long x_f = floor(x);
        long y_f = floor(y);
        long z_f = floor(z);
        const scalar_t wx = x - x_f;
        const scalar_t wy = y - y_f;
        const scalar_t wz = z - z_f;
        const scalar_t wxy = wx * wy;
        const scalar_t wxy_1 = (1 - wy) * (1 - wx);
        const scalar_t wz_1 = 1 - wz;
        const scalar_t wy_wxy = wy - wxy;
        const scalar_t wx_wxy = wx - wxy;
        for (int i = 0; i < 8; i++){
            long index_now = indices[b][n][j][i];
            for (int k = 0; k < F; k++){
                scalar_t grad_now = feature_grad[b][n][j * F + k];
                scalar_t * addr_now = reinterpret_cast<scalar_t *>(&hash_grad[b][j][index_now][k]);
                switch (i){
                    case 0:
                        gpuAtomicAdd(addr_now, grad_now * wz_1 * wxy_1);
                        break;
                    case 1:
                        gpuAtomicAdd(addr_now, grad_now * wz_1 * wy_wxy);
                        break;
                    case 2:
                        gpuAtomicAdd(addr_now, grad_now * wz_1 * wx_wxy);
                        break;
                    case 3:
                        gpuAtomicAdd(addr_now, grad_now * wz_1 * wxy);
                        break;
                    case 4:
                        gpuAtomicAdd(addr_now, grad_now * wz * wxy_1);
                        break;
                    case 5:
                        gpuAtomicAdd(addr_now, grad_now * wz * wy_wxy);
                        break;
                    case 6:
                        gpuAtomicAdd(addr_now, grad_now * wz * wx_wxy);
                        break;
                    case 7:
                        gpuAtomicAdd(addr_now, grad_now * wz * wxy);
                        break;
                }
            }
        }
    }
}


} // namespace end

torch::Tensor interp_3D_cuda_forward(
    torch::Tensor features, // B x N x 8 x F
    torch::Tensor coords // B x N x 3
){
    const int b = features.size(0);
    const int n = features.size(1); 
    const int f = features.size(3);
    auto interp_feats = torch::zeros({b,n,f},
                                    torch::TensorOptions().dtype(
                                       features.dtype()).device(
                                       features.device())
                                    );
    const int threads_num = 1024;
    const dim3 blocks_num((n + threads_num - 1) / threads_num, b);
    
    AT_DISPATCH_ALL_TYPES(features.type(), "interp_3D_cuda_forward", ([&] {
        interp_3D_forward_kernel<scalar_t><<<blocks_num, threads_num>>>(
            features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_feats.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));

    return interp_feats;
}

std::vector<torch::Tensor> retrieve_from_hash_table_cuda_forward(
    torch::Tensor hash_tables, // B x H_N x H_SIZE x F
    torch::Tensor coords, // B x N x 3
    const int res_min,
    const int res_max
){
    const int b = hash_tables.size(0);
    const int f = hash_tables.size(3);
    const int n = coords.size(1);
    const int dim = coords.size(2);
    const int table_num = hash_tables.size(1);

    auto interp_feats = torch::zeros({b, n, f*table_num},
                                     torch::TensorOptions().dtype(
                                        hash_tables.dtype()).device(
                                        hash_tables.device())
                                    );
    auto indices = torch::zeros({b, n, table_num, static_cast<int>(pow(2, dim))},
                                    torch::TensorOptions().dtype(
                                    torch::kLong).device(
                                    hash_tables.device())
                                );

    const int threads_num = 1024;
    const dim3 blocks_num((n + threads_num - 1) / threads_num, b);

    // Get the resolution list
    float resolution_list[table_num];
    float b_ = exp((log(res_max) - log(res_min)) / (table_num - 1));
    for (int i = 0; i < table_num; i++){
        resolution_list[i] = res_min * pow(b_, i);
    }
    float * res_list_device;
    cudaMalloc(&res_list_device, sizeof(float) * table_num);
    cudaMemcpy(res_list_device, resolution_list, sizeof(float) * table_num,
              cudaMemcpyHostToDevice);

    // run the kernels
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hash_tables.type(),
        "retrieve_hash_table_cuda_forward", ([&] {
        if (dim == 3){
            retrieve_from_hash_table_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
                hash_tables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                res_list_device,
                interp_feats.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>()
            );
        }
        else if (dim == 2) {
            retrieve_from_hash_table_2D_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
                hash_tables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                res_list_device,
                interp_feats.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>()
            );
        }
    }));

    cudaFree(res_list_device);

    return {interp_feats, indices};
}

std::vector<torch::Tensor> reconstruct_hash_table_cuda_forward(
    torch::Tensor feats, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x (H_N x F)
    const int table_dim,
    const int res_min,
    const int res_max
){
    const int b = feats.size(0);
    const int f = 2; // Only support entry dimension 2
    const int n = feats.size(1);
    const int table_num = feats.size(2) / f;
    const int dim = 2; // Now only support 2D

    auto recon_hash_tables = torch::zeros({b, table_num, table_dim, 2},
                                     torch::TensorOptions().dtype(
                                        feats.dtype()).device(
                                        feats.device())
                            );
    auto weights = torch::zeros({b, table_num, table_dim, 2},
                                     torch::TensorOptions().dtype(
                                        feats.dtype()).device(
                                        feats.device())
                            );
    auto indices = torch::zeros({b, n, table_num, static_cast<int>(pow(2, dim))},
                                    torch::TensorOptions().dtype(
                                    torch::kLong).device(
                                    feats.device())
                                );

    const int threads_num = 1024;
    const dim3 blocks_num((n + threads_num - 1) / threads_num, b);

    // Get the resolution list
    float resolution_list[table_num];
    float b_ = exp((log(res_max) - log(res_min)) / (table_num - 1));
    for (int i = 0; i < table_num; i++){
        resolution_list[i] = res_min * pow(b_, i);
    }
    float * res_list_device;
    cudaMalloc(&res_list_device, sizeof(float) * table_num);
    cudaMemcpy(res_list_device, resolution_list, sizeof(float) * table_num,
              cudaMemcpyHostToDevice);

    // run the kernels
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(),
        "reconstruct_hash_table_cuda_forward", ([&] {
        reconstruct_hash_table_2D_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
            feats.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            res_list_device,
            recon_hash_tables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>()
        );
    }));

    cudaFree(res_list_device);

    // Normalize by weight
    recon_hash_tables = recon_hash_tables / (weights + 1e-7);

    return {recon_hash_tables, weights, indices};
}


torch::Tensor retrieve_from_hash_table_cuda_backward(
    torch::Tensor feature_grad, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x 3
    torch::Tensor indices, // B x N x H_N x 8
    const int res_min,
    const int res_max,
    const int table_size
){
    const int b = coords.size(0);
    const int n = coords.size(1);
    const int dim = coords.size(2);
    const int table_num = indices.size(2);

    auto grad_table = torch::zeros({b, table_num, table_size, 2},
                                    torch::TensorOptions().dtype(
                                        feature_grad.dtype()).device(
                                        feature_grad.device())
                                    );

    const int threads_num = 1024;
    const dim3 blocks_num((n + threads_num - 1) / threads_num, b);

    // Get the resolution list
    float resolution_list[table_num];
    float b_ = exp((log(res_max) - log(res_min)) / (table_num - 1));
    for (int i = 0; i < table_num; i++){
        resolution_list[i] = res_min * pow(b_, i);
    }
    float * res_list_device;
    cudaMalloc(&res_list_device, sizeof(float) * table_num);
    cudaMemcpy(res_list_device, resolution_list, sizeof(float) * table_num,
              cudaMemcpyHostToDevice);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feature_grad.type(),
        "retrieve_hash_table_cuda_backward", ([&] {
        if (dim == 3){
            retrieve_from_hash_table_backward_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
                feature_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>(),
                res_list_device,
                grad_table.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
            );
        }
        else if (dim == 2){
            retrieve_from_hash_table_2D_backward_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
                feature_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>(),
                res_list_device,
                grad_table.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
            );
        }
    }
    ));

    cudaFree(res_list_device);

    return grad_table;
}


torch::Tensor reconstruct_hash_table_2D_cuda_backward(
    torch::Tensor table_grad, // B x H_N x H_S x F
    torch::Tensor coords, // B x N x 2
    torch::Tensor weights, // B x H_N x H_S x F
    torch::Tensor indices, // B x N x H_N x 4
    const int res_min,
    const int res_max,
    const int feat_size
){
    const int b = table_grad.size(0);
    const int n = feat_size;
    const int dim = 2;
    const int table_num = indices.size(2);

    auto feat_grad = torch::zeros({b, feat_size, table_num * 2},
                                    torch::TensorOptions().dtype(
                                        table_grad.dtype()).device(
                                        table_grad.device())
                                    );

    const int threads_num = 1024;
    const dim3 blocks_num((n + threads_num - 1) / threads_num, b);

    // Get the resolution list
    float resolution_list[table_num];
    float b_ = exp((log(res_max) - log(res_min)) / (table_num - 1));
    for (int i = 0; i < table_num; i++){
        resolution_list[i] = res_min * pow(b_, i);
    }
    float * res_list_device;
    cudaMalloc(&res_list_device, sizeof(float) * table_num);
    cudaMemcpy(res_list_device, resolution_list, sizeof(float) * table_num,
              cudaMemcpyHostToDevice);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(table_grad.type(),
        "reconstruct_hash_table_cuda_backward", ([&] {
        reconstruct_hash_table_2D_backward_kernel<scalar_t, 2><<<blocks_num, threads_num>>>(
            table_grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            coords.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<long, 4, torch::RestrictPtrTraits>(),
            res_list_device,
            feat_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }
    ));

    cudaFree(res_list_device);

    return feat_grad;
}