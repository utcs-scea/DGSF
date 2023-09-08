#ifndef _IO_H
#define _IO_H

#include <kmean.h>
#include <iostream>
#include <fstream>
#include <sstream>

void read_file_alloc_mem(struct kmean_t *kmean);

void write_file(struct kmean_t *kmean);

void print_centroids(float **centroids, int n_clusters, int n_dims);

void free_mem(struct kmean_t *kmean);

#endif
