#pragma once

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include "common.h"

struct options_t {
    int n_clusters;
    int dimensions;
    char *in_file;
    int max_iterations;
    real threshold;
    bool print_centroids;
    int seed;
    int algorithm;
};

void get_opts(int argc, char **argv, struct options_t *opts);
