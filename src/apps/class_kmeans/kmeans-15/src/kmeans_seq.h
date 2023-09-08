#pragma once

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <chrono>

double distance(int n_dims, double* input_vals, double* clusters);
void kmeans_seq(int n_vals,int n_dims,int* cluster_id, double** input_vals, int max_iter, double threshold, double*** clusters, int n_clusters );
