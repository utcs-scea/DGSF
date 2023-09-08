#ifndef _KMEANS_INIT_H_
#define _KMEANS_INIT_H_
#include "argparse.h"

float* init_centroids(struct options_t* opts, 
                      int               n_vals, 
                      float*            data);

#endif 