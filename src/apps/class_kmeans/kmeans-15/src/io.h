#ifndef _IO_H
#define _IO_H

#include <argparse.h>
#include <iostream>
#include <fstream>

void read_file(struct options_t* args,
               int*              n_vals,
               double***             input_vals,
               int*              n_dims);

//void write_file(struct options_t*         args,
//                struct prefix_sum_args_t* opts);

#endif
