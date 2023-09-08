#pragma once

#include <getopt.h>
#include <iostream>

struct options_t {
    int num_cluster;
    int dims;
    char* inputfilename;
    int max_num_iter;
    double threshold;
    bool output_centroids;
    unsigned int seed;
    bool cuda_shared;
    bool print_e2e;
};

void get_opts(int argc, char **argv, struct options_t *opts);