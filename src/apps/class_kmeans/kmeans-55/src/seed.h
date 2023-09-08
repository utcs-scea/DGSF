#pragma once

#include "common.h"

int k_means_rand();

void k_means_init_random_centroids(int n_points, int d, real *points,
    int k, real *centroids, int seed);
