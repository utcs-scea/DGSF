#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>

struct options_t {
    char *in_file;
    int n_clusters;
    int n_dims;
    int max_iter;
    unsigned long int seed;
    double threshold;
    bool control;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
