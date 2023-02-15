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

std::vector<torch::Tensor> reconstruct_hash_table_cuda_forward(
    torch::Tensor feats, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x 2
    int table_dim, // T
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

torch::Tensor reconstruct_hash_table_2D_cuda_backward(
    torch::Tensor table_grad, // B x H_N x H_S x F
    torch::Tensor coords, // B x N x 2
    torch::Tensor weights, // B x H_N x H_S x F
    torch::Tensor indices, // B x N x H_N x 4
    int res_min,
    int res_max,
    int feat_size
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

std::vector<torch::Tensor> reconstruct_hash_table_forward(
    torch::Tensor feats, // B x N x (H_N x F)
    torch::Tensor coords, // B x N x 2
    int table_dim, // T
    int res_min,
    int res_max
){
    CHECK_INPUT(feats);
    CHECK_INPUT(coords);
    return reconstruct_hash_table_cuda_forward(feats,
                                               coords,
                                               table_dim,
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

torch::Tensor reconstruct_hash_table_2D_backward(
    torch::Tensor table_grad, // B x H_N x H_S x F
    torch::Tensor coords, // B x N x 2
    torch::Tensor weights, // B x H_N x H_S x F
    torch::Tensor indices, // B x N x H_N x 4
    int res_min,
    int res_max,
    int feat_size
){
    CHECK_INPUT(table_grad);
    CHECK_INPUT(indices);
    return reconstruct_hash_table_2D_cuda_backward(table_grad,
                                                   coords,
                                                   weights,
                                                   indices,
                                                   res_min,
                                                   res_max,
                                                   feat_size);
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
    m.def("recon_forward", &reconstruct_hash_table_forward,
        "Reconstruct hash table from spatial features.\n"
        "Vars:\n"
        "  spatial_feats (torch.Tensor, CUDA ONLY, 4 dim): B x (H_N x F) x H x W\n"
        "  table_dim (int): table dimension, the T.\n"
        "  res_max (int): maximum resolution of the hash tables.\n"
        "  res_min (int): minimum resolution of the hash tables.\n"
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
    m.def("recon_backward", &reconstruct_hash_table_2D_backward,
        "Backward of the hash table recon\n"
        "Vars:\n"
        "  table_grad (torch.Tensor,CUDA ONLY, 4 dim): B x H_N x H_S x F.\n"
        "  indices (torch.Tensor, CUDA_ONLY, long, 4, dim): saved indices B x N x H_N x 4\n"
        "  res_min (int): lowest resolution of levels.\n"
        "  res_max (int): highest resolution of levels.\n"
        "  feat_size (int): one side size of the feat. The feat size will be B x C x feat_size x feat_size \n"
    );
}

