/*
    Retrieve and interp from hash table.
    Author: Dingdong Yang
    Contacts: dya62@sfu.ca
*/
#include <torch/extension.h>
#include <vector>
#include <iostream>

torch::Tensor interp_3D_cuda_forward(
    torch::Tensor features, // B x N x 8 x F
    torch::Tensor coords // B x N x 3
);

std::vector<torch::Tensor> retrieve_from_hash_table_cuda_forward(
    torch::Tensor hash_tables, // B x H_N x H_SIZE x F
    torch::Tensor coords, // B x N x 3
    int res_min,
    int res_max
);

torch::Tensor retrieve_from_hash_table_cuda_backward(
    torch::Tensor feature_grad, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x 3
    torch::Tensor indices, // B x N x H_N x 8
    int res_min,
    int res_max,
    int table_size
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor interp_3D_forward(
    torch::Tensor features, // B x N x 8 x F
    torch::Tensor coords // B x N x 3
){
    CHECK_INPUT(features);
    CHECK_INPUT(coords);
    return interp_3D_cuda_forward(features, coords);
}

std::vector<torch::Tensor> retrieve_from_hash_forward(
    torch::Tensor hash_tables, // B x H_N x H_SIZE x F
    torch::Tensor coords, // B x N x 3
    int res_min,
    int res_max
){
    CHECK_INPUT(hash_tables);
    CHECK_INPUT(coords);
    return retrieve_from_hash_table_cuda_forward(hash_tables,
                                                 coords,
                                                 res_min,
                                                 res_max);
}

torch::Tensor retrieve_from_hash_backward(
    torch::Tensor feature_grad, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x 3
    torch::Tensor indices, // B x N x H_N x 8
    int res_min,
    int res_max,
    int table_size
){
    CHECK_INPUT(feature_grad);
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    return retrieve_from_hash_table_cuda_backward(feature_grad,
                                                  coords,
                                                  indices,
                                                  res_min,
                                                  res_max,
                                                  table_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &interp_3D_forward, "3D interp forward.");
    m.def("retrieve_forward", &retrieve_from_hash_forward,
        "Instant-ngp retrieving forward.\n"
        "Vars:\n"
        "  hash_tables (torch.Tensor, CUDA ONLY, 4 dim): B x H_N x H_SIZE x F.\n"
        "  coords (torch.Tensor, CUDA ONLY, 3 dim): B x N x (3 or 2), value range (0, 1)\n"
        "  res_min (int): lowest resolution of levels.\n"
        "  res_max (int): highest resolution of levels.\n"
        "Returns:\n"
        "  sampled_features: B x N x (H_N) x F.\n"
        "  sampled_indices: B x N x H_N x (8 or 4).\n"
        );
    m.def("retrieve_backward", &retrieve_from_hash_backward,
        "Instant-ngp retrieving backward.\n"
        "Vars:\n"
        "  feature_grad (torch.Tensor, CUDA ONLY, 4 dim): B x N x (H_N x F).\n"
        "  coords (torch.Tensor, CUDA ONLY, 3 dim): B x N x (3 or 2), value range (0, 1)\n"
        "  indices (torch.Tensor, CUDA_ONLY, long, 4, dim): B x N x H_N x (8 or 4)\n"
        "  res_min (int): lowest resolution of levels.\n"
        "  res_max (int): highest resolution of levels.\n"
        "  table_size (int): \n"
        "Returns:\n"
        "  hash_table_grads (torch.Tensor, 4 dim): B x H_N x H_S x F"
        );
}

