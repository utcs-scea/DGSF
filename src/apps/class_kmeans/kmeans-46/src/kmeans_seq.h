#ifndef _KMEANS_SEQ_H_
#define _KMEANS_SEQ_H_

#include <stdlib.h>
#include <string.h>
#include <limits>
#include <chrono>
#include "argparse.h"

int find_min_dist_centroid(struct options_t* opts,
                           int               data_idx, 
                           float*            data,
                           float*            curr_centroids);

void kmeans_seq(float**           curr_centroids_p, 
                int*              iterations_p, 
                int*              microseconds_p, 
                struct options_t* opts, 
                float*            inputs_vals, 
                int               n_vals);

#endif 