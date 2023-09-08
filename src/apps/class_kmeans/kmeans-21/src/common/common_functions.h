#ifndef _COMMON_FUNCTIONS_H
#define _COMMON_FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

struct options_t {
    char *input_file_name;
    int num_cluster;
    int dims;
    int max_num_iter;
    double threshold;
    int seed;
    bool output_centroids;
    int shared_memory;
};

void get_opts(int argc, char **argv, struct options_t *opts);

void read_file(struct options_t *args, int *n_vals, float **points);

void get_initial_centroids(options_t *opts, int *n_nums, float *points, float **centroids);

void assign_clusters(options_t *opts, float *points, float *centroids, int *n_vals, int *cluster_mappings);

int assign_cluster(options_t *opts, float *point, int point_idx, float *centroids);

void recompute_centroids(options_t *opts, int *cluster_mappings, float *points, float *centroids, int *n_vals);

float get_squared_distance(int dims, float *point1, int s1, float *point2, int s2);

void print_cluster_centroids(options_t *opts, float *centroids);

void print_cluster_mappings(int *n_vals, int *cluster_mapping);

void swap_centroids(float **old_centroids, float **centroids);

bool test_convergence(options_t *opts, float *old_centroids, float *centroids);

bool test_convergence(options_t *opts, float *centroid_distances);

#endif
