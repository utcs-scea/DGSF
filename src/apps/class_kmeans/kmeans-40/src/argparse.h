#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    char *in_file;
    int num_cluster;
    int num_points;
    int dims;
    int max_num_iter;
    double threshold;
    int seed;
    bool control;
    bool end2end;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
