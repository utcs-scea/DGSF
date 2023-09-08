#pragma once

#include "common.h"

int k_means_sequential(int n_points, real *points, struct options_t *opts,
    int* point_cluster_ids, real** centroids);
