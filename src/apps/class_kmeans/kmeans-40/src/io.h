#ifndef _IO_H
#define _IO_H

#include <argparse.h>
#include <iostream>
#include <fstream>

typedef float REAL;

void read_file(struct options_t* args,
               REAL**            input_dataset,
               int**             output_labels,
               REAL**            output_centroids);
               
#endif
