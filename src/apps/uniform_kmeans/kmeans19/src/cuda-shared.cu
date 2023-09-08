#include <chrono>
#include <cmath>
#include <iostream>
#include <functional>
#include <limits>
#include <vector>

#include "common.h"


#define check_cuda_error(val) { check_error((val), __FILE__, __LINE__); }
inline void check_error(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(
            stderr,
            "CUDA Error: %s %s %d\n",
            cudaGetErrorString(code),
            file,
            line);
    }
}

__global__ void find_nearest_centroids_(
        const float *points,
        const float *centroids,
        int *labels,
        const int dims,
        const int num_cluster)
{
    extern __shared__ float distances[];
    distances[threadIdx.x] = 0.0;
    __syncthreads();

    for (int dim = 0; dim < dims; ++dim) {
        float dist =
            points[blockIdx.x * dims + dim] -
            centroids[threadIdx.x * dims + dim];
        distances[threadIdx.x] += dist * dist;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float min_dist = 999999999;
        int label = -1;
        for (int i = 0; i < num_cluster; ++i) {
            if (distances[i] < min_dist) {
                label = i;
                min_dist = distances[i];
            }
        }

        labels[blockIdx.x] = label;
    }
}

void find_nearest_centroids(
    const float *dev_points,
    const int dev_points_size,
    const float *dev_centroids,
    int *dev_labels,
    const options_t &opts)
{
    find_nearest_centroids_<<<
        dev_points_size / opts.dims,
        opts.num_cluster,
        opts.num_cluster * sizeof(float)>>>
    (
        dev_points,
        dev_centroids,
        dev_labels,
        opts.dims,
        opts.num_cluster
    );
}

__global__ void update_centroids_(
    const float *points,
    float *centroids,
    const int *labels,
    int labels_size,
    int dims)
{
    int count = 0;
    float newVal = 0.0;

    for (int i = 0; i < labels_size; ++i) {
        if (labels[i] == blockIdx.x) {
            ++count;
            newVal += points[i * dims + threadIdx.x];
        }
    }

    if (count > 0) {
        centroids[blockIdx.x * dims + threadIdx.x] = newVal / count;
    } else {
        centroids[blockIdx.x * dims + threadIdx.x] = 0;
    }
}

void update_centroids(
    const float *dev_points,
    float *dev_centroids,
    const int dev_centroids_size,
    const int *dev_labels,
    const int dev_labels_size,
    const options_t &opts)
{
    update_centroids_<<<dev_centroids_size / opts.dims, opts.dims>>>(
        dev_points,
        dev_centroids,
        dev_labels,
        dev_labels_size,
        opts.dims);
}

__global__ void converged_(
    const float *old_centroids,
    const float *centroids,
    bool *converged,
    const float threshold)
{
    __shared__ float distance;
    distance = 0.0;
    __syncthreads();

    float dist = centroids[blockIdx.x + threadIdx.x] - old_centroids[blockIdx.x + threadIdx.x];
    atomicAdd(&distance, dist * dist);
    __syncthreads();

    if (threadIdx.x == 0 && sqrt(distance) > threshold) {
        converged[0] = false;
    }
}

bool converged(
    const float *dev_old_centroids,
    const float *dev_centroids,
    const int dev_centroids_size,
    const options_t &opts)
{
    bool *dev_converged;
    cudaMalloc((void**) &dev_converged, sizeof(bool));
    cudaMemset(dev_converged, 1, sizeof(bool));

    converged_<<<dev_centroids_size / opts.dims, opts.dims>>>(
        dev_old_centroids, dev_centroids, dev_converged, opts.threshold);

    bool converged;
    cudaMemcpy(&converged, dev_converged, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(dev_converged);
    return converged;
}

int main(int argc, char *argv[])
{
    struct options_t opts;
    get_opts(argc, argv, &opts);

    std::vector<float> points;
    float *dev_points;
    int n_points;
    read_file(opts, n_points, points);

    check_cuda_error(cudaMalloc(
        (void **) &dev_points,
        points.size() * sizeof(float)));

    check_cuda_error(cudaMemcpy(
        dev_points,
        points.data(),
        points.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<float> centroids(opts.num_cluster * opts.dims);
    generate_centroids(points, centroids, opts);
    float *dev_centroids;
    float *dev_old_centroids;
    int *dev_labels;

    check_cuda_error(cudaMalloc(
        (void **) &dev_centroids,
        centroids.size() * sizeof(float)));

    check_cuda_error(cudaMemcpy(
        dev_centroids,
        centroids.data(),
        centroids.size() * sizeof(float), cudaMemcpyHostToDevice));

    check_cuda_error(cudaMalloc(
        (void **) &dev_old_centroids,
        centroids.size() * sizeof(float)));

    check_cuda_error(cudaMalloc((void **) &dev_labels, n_points * sizeof(int)));

    // Start timer
	// Start -> Stop Events used to record time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int i = 0;
    while (++i < opts.max_num_iter) {
        cudaMemcpy(
            dev_old_centroids,
            dev_centroids,
            centroids.size() * sizeof(float), cudaMemcpyDeviceToDevice);

        find_nearest_centroids(
            dev_points,
            points.size(),
            dev_centroids,
            dev_labels,
            opts);

        update_centroids(
            dev_points,
            dev_centroids,
            centroids.size(),
            dev_labels,
            n_points,
            opts);

        bool done = converged(
            dev_old_centroids, dev_centroids, centroids.size(), opts);

        if (done) {
            break;
        }
    }

    //End timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time_in_ms = 0;
    cudaEventElapsedTime(&total_time_in_ms, start, stop);

    check_cuda_error(cudaMemcpy(
        centroids.data(),
        dev_centroids,
        centroids.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    std::vector<int> labels(n_points);

    check_cuda_error(cudaMemcpy(
        labels.data(),
        dev_labels,
        labels.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    output_results(
        labels,
        centroids,
        total_time_in_ms,
        i,
        opts);

    cudaFree(dev_points);
    cudaFree(dev_centroids);
    cudaFree(dev_old_centroids);
    cudaFree(dev_labels);

    return 0;
}