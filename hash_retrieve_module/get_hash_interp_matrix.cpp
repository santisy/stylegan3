/*
    Get the matrix for hash table resizing
    Author: Dingdong Yang
    Contacts: dya62@sfu.ca
*/

#include <torch/extension.h>
#include <vector>
#include <iostream>


torch::Tensor get_hash_interp_matrix_2D(int input_size,
                                        int table_num,
                                        int res_min,
                                        int res_max,
                                        int up_scale
                                        ){
    /*
        Args:
            input_size (int): input table size
            table_num (int): table_number
            res_min (int): minimum resolution
            res_max (int): maximum resolution
            up_scale (int): upsample scale
    */
    int output_size = input_size * up_scale;
    auto matrix_out = torch::zeros({table_num, output_size, input_size});
    auto m = matrix_out.accessor<float, 3>();
    float b_ = exp((log(res_max) - log(res_min)) / (table_num - 1));

    for (int k = 0; k < table_num; k++){
        long res_now = floor(res_min * pow(b_, k));
        for (int i = 0; i < res_max; i ++){
            for (int j = 0; j < res_max; j++){
                long x = floor((i + 0.5f) / res_max * res_now);
                long y = floor((j + 0.5f) / res_max * res_now);
                long index_input = (x ^ (y * 2654435761)) % (input_size / 2);
                long index_output = (x ^ (y * 2654435761)) % (output_size / 2);
                m[k][index_output * 2][index_input * 2] += 1.f;
                m[k][index_output * 2 + 1][index_input * 2 + 1] += 1.f;
            }
        }
    }

    auto sum_ = torch::sum(matrix_out, {2}, true);
    matrix_out = matrix_out / sum_;

    return matrix_out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("get_hash_interp_matrix_2D", &get_hash_interp_matrix_2D,
        "Get hash interp matrix 2D\n"
        "Vars:\n"
        "   input_size (int): input table_size.\n"
        "   table_num (int): number of tables (tokens).\n"
        "   res_min(int): minimum resolution.\n"
        "   res_max(int): maximum resolution.\n"
        "   up_scale(int): upsample scale\n"
        );
}

