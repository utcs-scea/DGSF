#pragma once

#include "common.h"
#include "argparse.h"

int k_means_cuda(int n_points, real *points, struct options_t *opts,
    int* point_cluster_ids, real** centroids, double* per_iteration_time);
