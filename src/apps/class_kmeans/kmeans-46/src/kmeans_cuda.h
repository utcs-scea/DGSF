#ifndef _KMEANS_CUDA_H_
#define _KMEANS_CUDA_H_

#include <string.h>
#include <chrono>
#include <cfloat>
#include "argparse.h"

void printCudaInfo();

void kmeans_cuda_basic(float**           curr_centroids_p, 
					   int*              iterations_p, 
					   int*              copy_milliseconds_p,
					   int*              exec_milliseconds_p, 
					   struct options_t* h_opts, 
					   float*            input_vals, 
					   int               n_vals);

void kmeans_cuda_shmem(float**           centroids_p, 
					   int*              iterations_p, 
					   int*              copy_milliseconds_p,
					   int*              exec_milliseconds_p, 
					   struct options_t* h_opts, 
					   float*            h_input_vals, 
					   int               n_vals);

#endif 