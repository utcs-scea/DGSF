#include <random>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <iterator>
#include <cuda.h>
#include <device_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "./lib/argparse.h"
#include "./lib/datasets.h"
#include "kmeans_kernels.h"
#include "./lib/timer.h"

#include <iostream>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

struct kmeans_t {
    std::vector<double> centroids;
    std::vector<int> labels;
};


bool converged(const std::vector<double>& a, const std::vector<double>&  b, double threshold, int num_cluster, int dims) {
    for (int i = 0; i < num_cluster; ++i) {
        double dist = 0.0;
        for (int j = 0; j < dims; j++) {
            double x = a[i * dims + j] - b[i * dims + j];
            dist += x * x;
        }
        // Compare squared distance
        if (dist > threshold * threshold) {
            return false;
        }
    }
    return true;
}


kmeans_t* kmeans(cdataset_t* data, args_t* args) {
    kmeans_t* res = new kmeans_t;
    res->labels.resize(data->size);
    // Initialize centroids
    std::vector<double> init_centroids((size_t) data->num_cluster * data->dims);
    for (int i = 0; i < data->num_cluster; i++) {
        size_t index = kmeans_rand() % data->size;
        std::copy(data->points.cbegin() + (index * data->dims), data->points.cbegin() + ((index + 1) * data->dims), init_centroids.begin() + (i * data->dims));
    }
    res->centroids = init_centroids;
    int iterations = 0;

    size_t data_sz = sizeof(double) * data->size * data->dims;
    double* d_data = NULL;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&d_data, data_sz);
    err = cudaMemcpy(d_data, &data->points[0], data_sz, cudaMemcpyHostToDevice);

    double* d_centroids = NULL;
    size_t centroids_sz = sizeof(double) * data->num_cluster * data->dims;
    err = cudaMalloc((void**)&d_centroids, centroids_sz);
    err = cudaMemcpy(d_centroids, &res->centroids[0], centroids_sz, cudaMemcpyHostToDevice);

    double* d_distances = NULL;
    size_t distances_sz = sizeof(double) * data->num_cluster * data->size;
    err = cudaMalloc((void**)&d_distances, distances_sz);

    int* d_labels = NULL;
    size_t labels_sz = sizeof(int) * data->size;
    err = cudaMalloc((void**)&d_labels, labels_sz);

    int* d_counts = NULL;
    size_t counts_sz = sizeof(int) * data->num_cluster * data->dims;
    err = cudaMalloc((void**)&d_counts, counts_sz);
    Timer t;
    std::vector<double> new_centroids = res->centroids;
    do {
        res->centroids = new_centroids;

        kcalcDistanceShared(d_data, d_centroids, d_distances, data->size, data->num_cluster, data->dims);

        knaiiveArgmin(d_distances, d_labels, data->size, data->num_cluster);

        kcalcNewCentroids(d_data, d_labels, d_centroids, d_counts, data->size, data->dims, data->num_cluster);

        cudaMemset(d_distances, 0, distances_sz);

        cudaMemcpy(&new_centroids[0], d_centroids, centroids_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(&res->labels[0], d_labels, labels_sz, cudaMemcpyDeviceToHost);
    } while (iterations++ < args->max_num_iter && !converged(res->centroids, new_centroids, args->threshold, data->num_cluster, data->dims));
    std::cout << iterations << "," << t.elapsed() / iterations << std::endl;
    res->centroids = new_centroids;
    return res;
}

int main(int argc, char* argv[]) {
    args_t* args = parse_arguments(argc, argv);
    cdataset_t* dataset = load_cdataset(args);
    kmeans_srand(args->seed);
    kmeans_t* res = kmeans(dataset, args);

    if (args->output_centroids) {
        for (int clusterId = 0; clusterId < dataset->num_cluster; clusterId++) {
            std::cout << clusterId << " ";
            for (int d = 0; d < dataset->dims; d++)
                std::cout << res->centroids[clusterId * dataset->dims + d] << " ";
            std::cout << std::endl;
        }
    }
    else {
        std::cout << "clusters:";
        for (int idx = 0; idx < dataset->size; idx ++) {
            std::cout << " " << res->labels[idx];
        }
    }
    return 0;
}
