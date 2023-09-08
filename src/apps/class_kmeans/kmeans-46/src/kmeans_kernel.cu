include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

void find_centroid_thrust(struct options_t* opts, int data_idx, int n_vals, double* data, double* centroids, int* centroid_cnts) { 
    double min_sq_dist = numeric_limits<double>::max();
    int min_cluster = 0;
    for (int i = 0; i < opts->k_clusters; i++) {
        double* data_pt = data + (data_idx * opts->n_dims);
        double* centroid_pt = centroids + (i * opts->n_dims);
        double cluster_sq_dist = f(opts->n_dims, data_pt, centroid_pt);
        if (cluster_sq_dist < min_sq_dist) { 
            min_sq_dist = cluster_sq_dist; 
            min_cluster = i;
        }
    }
    for (int i = 0; i < opts->n_dims; i++) {
        centroids[(min_cluster * opts->n_dims) + i] += data[(data_idx * opts->n_dims) + i];
    }
    centroid_cnts[min_cluster]++;
}

void kmeans_thrust()
{
    thrust::host_vector<int> h_vec(n_vals);
    thrust::device_vector<int> d_vec = h_vec;
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    return 0;
}