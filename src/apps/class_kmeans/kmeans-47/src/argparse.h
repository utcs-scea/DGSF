//
//  argparse.h
//  PS_Lab2_Test
//
//  Created by Heejong Jang O'Keefe on 11/11/20.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//

#ifndef argparse_h
#define argparse_h

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
   int n_cluster;
   int n_dims;
   char *in_file;
   int max_iter;
   float threshold;
   bool centroid;
   int seed;
};

void get_opts(int argc, char **argv, struct options_t *opts);

#endif /* argparse_h */
