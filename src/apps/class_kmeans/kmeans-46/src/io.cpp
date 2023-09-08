#include "io.h"

void read_file(int*              n_vals_p,
               float**           input_vals_p,
			   struct options_t* opts) {

    std::fstream file(opts->in_file, std::ios_base::in);

    float* input_vals;
    int n_vals; 
    
    int line_idx = 0;
    int data_idx = 0;
    int dim_idx  = 0;
    
    int   i_val;
    float f_val;
    
    std::string str;
    while(getline(file, str)) {
        if (line_idx == 0) {
            std::istringstream ss(str);
            ss >> n_vals;
            input_vals = (float*)malloc(n_vals * opts->n_dims * sizeof(float));
         } else if (line_idx > n_vals) {
            break;
        } else { 
            std::istringstream ss(str);
            ss >> i_val;
            data_idx = line_idx - 1;
            assert(i_val-1 == data_idx);
            dim_idx = 0;
            while (ss >> f_val) { 
                input_vals[(data_idx * opts->n_dims) + dim_idx] = f_val;
                dim_idx++;
            }
             assert(dim_idx == opts->n_dims);
         }
         line_idx++;
    }
    *n_vals_p     = n_vals;
    *input_vals_p = input_vals;
}