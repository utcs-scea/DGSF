#pragma once

#include <thrust/functional.h>

#include "common.h"
#include "argparse.h"

int k_means_thrust(int n_points, real *points, struct options_t *opts,
    int* point_cluster_ids, real** centroids, double* per_iteration_time);
