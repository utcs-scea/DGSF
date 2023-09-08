#include <cuda.h>
#include <iostream>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}


#endif

__global__ void dot(double* points, double* centroids, int dims, int npoints, int ncentroids, int points_size, int centroids_size, int cross_size, double* distances) {
    int points_per_block = npoints / gridDim.x;
    int start_point_idx = blockIdx.x * points_per_block;
    int centroid_idx = threadIdx.x;
    
    for (int i = start_point_idx; i < start_point_idx + points_per_block && i < npoints; i++) {
        int distance_idx = i * ncentroids + centroid_idx;
        if (distance_idx >= cross_size) {
            return;
        }
        distances[distance_idx] = 0;
        for (int j = 0; j < dims; j++) {
            int point_value_idx = i * dims + j;
            int centroid_value_idx = centroid_idx * dims + j;
            if (point_value_idx < points_size && centroid_value_idx < centroids_size) {
                double diff = points[point_value_idx] - centroids[centroid_value_idx];
                distances[distance_idx] += diff * diff;
            }
        }
    }
}

__global__ void sqrt_kernel(double* values, int n) {
    int values_per_thread = n / gridDim.x / blockDim.x;
    int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread;
    
    for (int i = start_idx; i < start_idx + values_per_thread && i < n; i++) {
        values[i] = sqrt(values[i]);
    }
}

__global__ void nearest_centroid(double* distances, int npoints, int ncentroids, int cross_size, int* nearest_centroids) {
    int points_per_thread = npoints / gridDim.x / blockDim.x;
    int start_point_idx = (blockIdx.x * blockDim.x + threadIdx.x) * points_per_thread;
    
    double minimum = 2e9;
    for (int i = start_point_idx; i < start_point_idx + points_per_thread && i < npoints; i++) {
        for (int j = 0; j < ncentroids; j++) {
            int idx = i * ncentroids + j;
            if (idx < cross_size && distances[idx] < minimum) {
                minimum = distances[idx];
                nearest_centroids[i] = j;
            }
        }
    }
}

__global__ void count_centroid_id(int* nearest_centroids, int npoints, int ncentroids, int* counts) {
    int points_per_thread = npoints / gridDim.x / blockDim.x;
    int start_point_idx = (blockIdx.x * blockDim.x + threadIdx.x) * points_per_thread;
    
    for (int i = start_point_idx; i < start_point_idx + points_per_thread && i < npoints; i++) {
        if (nearest_centroids[i] < ncentroids) {
            atomicAdd(counts + nearest_centroids[i], 1);
        }
    }
}

__global__ void sum_new_centroid_values(double* points, int* nearest_centroids, int dims, int npoints, int points_size, int centroids_size, double* new_centroid_values) {
    int points_per_block = npoints / gridDim.x;
    int start_point_idx = blockIdx.x * points_per_block;
    int d = threadIdx.x;
    
    for (int i = start_point_idx; i < start_point_idx + points_per_block && i < npoints; i++) {
        int centroid_id = nearest_centroids[i];
        int new_centroid_value_idx = centroid_id * dims + d;
        int point_value_idx = i * dims + d;
        if (centroid_id < npoints && new_centroid_value_idx < centroids_size && point_value_idx < points_size) {
            atomicAdd(new_centroid_values + new_centroid_value_idx, points[point_value_idx]);
        }
    }
}

__global__ void avg_new_centroid_values(double* new_centroid_values, int* counts, int dims, int ncentroids, int centroids_size) {
    int centroids_per_block = ncentroids / gridDim.x;
    int start_centroid_idx = blockIdx.x * centroids_per_block;
    int d = threadIdx.x;
    
    for (int i = start_centroid_idx; i < start_centroid_idx + centroids_per_block && i < ncentroids; i++) {
        int centroid_value_idx = i * dims + d;
        if (centroid_value_idx < centroids_size) {
            new_centroid_values[centroid_value_idx] /= counts[i];
        }
    }
}

__global__ void new_centroid_movement_squared(double* centroids, double* new_centroids, int dims, int ncentroids, int centroids_size, double* distances) {
    int centroids_per_block = ncentroids / gridDim.x;
    int start_centroid_idx = blockIdx.x * centroids_per_block;
    int d = threadIdx.x;
    
    for (int i = start_centroid_idx; i < start_centroid_idx + centroids_per_block && i < ncentroids; i++) {
        int centroid_value_idx = i * dims + d;
        if (centroid_value_idx < centroids_size) {
            double diff = centroids[i * dims + d] - new_centroids[i * dims + d];
            atomicAdd(distances + i, diff * diff);            
        }
    }
}

__global__ void is_convergent(double* distances, double threshold, int ncentroids, int* ret) {
    int values_per_thread = ncentroids / gridDim.x / blockDim.x;
    int start_value_idx = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread;
    
    for (int i = start_value_idx; i < start_value_idx + values_per_thread && i < ncentroids; i++) {
        if (distances[i] < threshold) {
            atomicAdd(ret, 1);
        }
        distances[i] = 0.0;
    }
}

__global__ void reset_zero(double* array, int n) {
    int values_per_thread = n / gridDim.x / blockDim.x;
    int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread;
    for (int i = start_idx; i < start_idx + values_per_thread && i < n; i++) {
        array[i] = 0.0;
    }
}

__global__ void nearest_centroid_shared(double* points, double* centroids, int dims, int npoints, int ncentroids, int points_size, int centroids_size, int cross_size, int* nearest_centroids) {
    int points_per_block = npoints / gridDim.x;
    int start_point_idx = blockIdx.x * points_per_block;
    int centroid_idx = threadIdx.x;
    
    __shared__ double s_distances[1024];
    __shared__ double s_centroids[2048];
    __shared__ int s_nearest_centroids[16];
    
    for (int i = 0; i < dims; i++) {
        s_centroids[centroid_idx * dims + i] = centroids[centroid_idx * dims + i];
    }
    
    for (int i = start_point_idx; i < start_point_idx + points_per_block && i < npoints; i++) {
        int distance_idx = (i - start_point_idx) * ncentroids + centroid_idx;
        if (distance_idx >= cross_size) {
            return;
        }
        s_distances[distance_idx] = 0;
        for (int j = 0; j < dims; j++) {
            int point_value_idx = i * dims + j;
            int centroid_value_idx = centroid_idx * dims + j;
            if (point_value_idx < points_size && centroid_value_idx < centroids_size) {
                double diff = points[point_value_idx] - s_centroids[centroid_value_idx];
                s_distances[distance_idx] += diff * diff;
            }
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        double minimum = 2e9;
        for (int i = start_point_idx; i < start_point_idx + points_per_block && i < npoints; i++) {
            for (int j = 0; j < ncentroids; j++) {
                int idx = (i - start_point_idx) * ncentroids + j;
                if (idx < cross_size && s_distances[idx] < minimum) {
                    minimum = s_distances[idx];
                    s_nearest_centroids[i - start_point_idx] = j;
                }
            }
            nearest_centroids[i] = s_nearest_centroids[i - start_point_idx];
        }
    }
}
