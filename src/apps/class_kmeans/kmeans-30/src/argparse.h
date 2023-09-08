#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <kmean.h>


void get_opts(int argc, char **argv, struct kmean_t *kmean);
#endif