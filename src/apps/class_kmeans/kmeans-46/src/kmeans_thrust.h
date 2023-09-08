#ifndef _KMEANS_THRUST_H_
#define _KMEANS_THRUST_H_

#include <stdlib.h>
#include <string.h>
#include <limits>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include "argparse.h"

void kmeans_thrust(double**          centroids_p, 
				   int*              iterations_p, 
				   int*              copy_milliseconds_p,
				   int*              exec_milliseconds_p, 
				   struct options_t* h_opts, 
				   double*           h_input_vals, 
				   int               n_vals); 

#endif 