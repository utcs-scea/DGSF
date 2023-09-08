#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int k_clusters;
    int n_dims; 
    const char* in_file;
    int max_num_iters;
    float threshold;
    bool print_copy_time;
	bool print_total_time;
    bool output_centroids;
    int seed;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
