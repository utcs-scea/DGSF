#ifndef _KMEANS_H
#define _KMEANS_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int num_cluster;
    int dims;
    char *inputfilename;
    int max_num_iter;
    double threshold;
    bool centroid;
    int seed;
};

int kmeans(const struct options_t& opts,
            int n_vals,
            double* centroid,
            double* input_vals,
            int* clusterId_of_point,
            float* elapsed_time,
            float* data_transfer_time);

#endif