#ifndef _IO_H_
#define _IO_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <getopt.h>
#include "argparse.h"

void read_file(int*     n_vals,
               float**  input_vals_p,
               struct   options_t* opts);

#endif 