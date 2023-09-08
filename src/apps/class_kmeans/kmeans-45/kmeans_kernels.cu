#include <iostream>
#include <stdio.h>

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Each thread will be responsible for computing the argmax of one point, this assumes n_centroids is a reasonable size
__global__ void naiiveArgmin(const double* distances, int* labels, int n_points, int n_centroids) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int distances_i = i * n_centroids;

    if (i < n_points) {
        int minIdx = 0;
        double minDist = distances[distances_i];
        for (int j = 0; j < n_centroids; j++) {
            double val = distances[distances_i + j];
            if (val < minDist) {
                minDist = val;
                minIdx = j;
            }
        }
        //printf("i: %d, %d, label: %d\n", i, distances_i, minIdx);
        labels[i] = minIdx;
    }
}

__global__ void
calcDistance(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int data_i = (i / (dims * n_centroids));
    int off = i % dims;
    int centroid_i = (i / dims) % n_centroids;
    int output_i = (data_i * n_centroids) + centroid_i;

    if (data_i < n_points)
    {
        //printf("i: %d, data_i: %d, off: %d, centroid_i: %d, output_i: %d, val:%lf\n", i, data_i, off, centroid_i, output_i, (centroids[centroid_i * dims + off] - data[data_i * dims + off]) * (centroids[centroid_i * dims + off] - data[data_i * dims + off]));
        atomicAdd(&output[output_i], (centroids[centroid_i * dims + off] - data[data_i * dims + off]) * (centroids[centroid_i * dims + off] - data[data_i * dims + off]));
    }
}

// Like calcDistances, but centroids are loaded into shared memory to avoid repeated reads
__global__ void
calcDistanceShared(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims)
{
    extern __shared__ double* s_centroids;

    for (int idx = 0; idx  < n_centroids * dims; idx += blockDim.x) {
        s_centroids[idx] = centroids[idx];
    }
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int data_i = (i / (dims * n_centroids));
    int off = i % dims;
    int centroid_i = (i / dims) % n_centroids;
    int output_i = (data_i * n_centroids) + centroid_i;

    if (data_i < n_points)
    {
        atomicAdd(&output[output_i], (s_centroids[centroid_i * dims + off] - data[data_i * dims + off]) * (s_centroids[centroid_i * dims + off] - data[data_i * dims + off]));
    }
}

__global__ void
calcNewCentroidsSum(const double* data, const int* labels, double* newCentroids, int* counts, int n_points, int dims)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int labels_i = i / dims;
    int centroid_i = labels[labels_i] * dims + i % dims;

    if (i < n_points * dims)
    {
        atomicAdd(&newCentroids[centroid_i], data[i]);
        atomicAdd(&counts[centroid_i], 1);
    }
}

__global__ void
divideVector(double* data, const int* counts, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        data[i] = data[i] / (double) counts[i];
    }
}

__global__ void
converged(const double* distances, int* converged, double threshold, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        if (distances[i] > threshold) {
            atomicAdd(converged, 1);
        }
    }
}

__global__ void
centroidChanged(const double* oldCentroids, const double* newCentroids, double* distances, int n, int dims)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int centroid_i = i / dims;

    if (centroid_i < n) {
        atomicAdd(&distances[centroid_i], (oldCentroids[i] - newCentroids[i]) * (oldCentroids[i] - newCentroids[i]));
    }
}



void kcalcDistance(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (n_points * n_centroids * dims + threadsPerBlock - 1) / threadsPerBlock;
    calcDistance << <blocksPerGrid, threadsPerBlock >> > (data, centroids, output, n_points, n_centroids, dims);

    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;
}

void kcalcDistanceShared(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n_points * n_centroids * dims + threadsPerBlock - 1) / threadsPerBlock;
    calcDistance << <blocksPerGrid, threadsPerBlock, sizeof(double) * n_centroids * dims >> > (data, centroids, output, n_points, n_centroids, dims);
}

void knaiiveArgmin(const double* distances, int* labels, int n_points, int n_centroids) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_points + threadsPerBlock - 1) / threadsPerBlock;
    naiiveArgmin << <blocksPerGrid, threadsPerBlock >> > (distances, labels, n_points, n_centroids);
}


void kcalcNewCentroids(const double* data, const int* labels, double* newCentroids, int* counts, int n_points, int dims, int n_centroids) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (n_points * dims + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(newCentroids, 0, sizeof(double) * n_centroids * dims);
    cudaMemset(counts, 0, sizeof(int) * n_centroids * dims);

    calcNewCentroidsSum << <blocksPerGrid, threadsPerBlock >> > (data, labels, newCentroids, counts, n_points, dims);

    int divideBlocksPerGrid = (n_centroids * dims  + threadsPerBlock - 1) / threadsPerBlock;

    divideVector << <blocksPerGrid, threadsPerBlock >> > (newCentroids, counts, n_centroids * dims);
}

void kConverged(const double* oldCentroids, const double* newCentroids, int* c, double* distances, int dims, int n_centroids, double threshold) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (n_centroids * dims + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(c, 0, sizeof(int));

    centroidChanged << <blocksPerGrid, threadsPerBlock >> > (oldCentroids, newCentroids, distances, n_centroids, dims);
    blocksPerGrid = (n_centroids + threadsPerBlock - 1) / threadsPerBlock;
    converged << <blocksPerGrid, threadsPerBlock >> > (distances, c, threshold * threshold, n_centroids);
}