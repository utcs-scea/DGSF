#ifndef _IO_H
#define _IO_H

//#include <argparse.h>
#include "argparse.h"
#include <iostream>
#include <fstream>
#include <vector>

//void read_file(struct options_t* args,
//               int*              n_vals,
//               float** input_vals);

 void read_file(struct options_t* args,
                int*              n_vals,
                std::vector<std::vector<float>>& input_vals);

// void read_file(struct options_t* args,
//                int*              n_vals,
//                int**             input_vals,
//                int**             output_vals);

#endif
