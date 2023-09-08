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
        float *distances,
        const int dims,
        const int num_cluster)
{
    for (int centroid_id = 0; centroid_id < num_cluster; ++centroid_id) {
        float dist =
            points[blockIdx.x * blockDim.x + threadIdx.x] -
            centroids[centroid_id * dims + threadIdx.x % dims];

        atomicAdd(
            &distances[blockIdx.x * num_cluster + centroid_id],
            dist * dist);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float min_dist = 999999999;
        int label = -1;
        int idx_start = blockIdx.x * num_cluster;
        for (int i = idx_start; i < idx_start + num_cluster; ++i) {
            if (distances[i] < min_dist) {
                label = i % num_cluster;
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
    float *distances,
    const options_t &opts)
{
    
    cudaMemset(
        distances,
        0,
        dev_points_size / opts.dims * opts.num_cluster * sizeof(float));

    find_nearest_centroids_<<<dev_points_size / opts.dims, opts.dims>>>(
        dev_points,
        dev_centroids,
        dev_labels,
        distances,
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

__global__ void get_dist(
    const float *old_centroids,
    const float *centroids,
    float *distances)
{
    float dist = centroids[blockIdx.x] - old_centroids[blockIdx.x];
    distances[blockIdx.x] = dist * dist;
}

__global__ void converged_(
    const float *distances,
    bool *converged,
    const int dims,
    const float threshold)
{
    float distance = 0.0;
    for (int i = blockIdx.x * dims; i < blockIdx.x * dims + dims; ++i) {
        distance += distances[i];
    }

    if (sqrt(distance) > threshold) {
        converged[0] = false;
    }
}

bool converged(
    const float *dev_old_centroids,
    const float *dev_centroids,
    const int dev_centroids_size,
    float *distances,
    bool *dev_converged,
    const options_t &opts)
{
    cudaMemset(distances, 0, dev_centroids_size * sizeof(float));

    get_dist<<<dev_centroids_size, 1>>>(
        dev_old_centroids, dev_centroids, distances);

    cudaMemset(dev_converged, 1, sizeof(bool));

    converged_<<<dev_centroids_size / opts.dims, 1>>>(
        distances, dev_converged, opts.dims, opts.threshold);

    bool converged;
    cudaMemcpy(&converged, dev_converged, sizeof(bool), cudaMemcpyDeviceToHost);
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

    float *distances;
    cudaMalloc(
        (void **) &distances,
        points.size() / opts.dims * opts.num_cluster * sizeof(float));

    float *converged_distances;
    cudaMalloc((void **) &converged_distances, centroids.size() * sizeof(float));

    bool *dev_converged;
    cudaMalloc((void**) &dev_converged, sizeof(bool));

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
            distances,
            opts);

        update_centroids(
            dev_points,
            dev_centroids,
            centroids.size(),
            dev_labels,
            n_points,
            opts);

        bool done = converged(
            dev_old_centroids, dev_centroids, centroids.size(), 
            converged_distances, dev_converged, opts);

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
    cudaFree(distances);
    cudaFree(dev_converged);
    cudaFree(converged_distances);

    return 0;
}
