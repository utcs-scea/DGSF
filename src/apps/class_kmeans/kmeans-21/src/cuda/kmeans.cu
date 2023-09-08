#include <cuda_runtime.h>
#include "../common/common_functions.h"
#include "kmeans.h"

__host__ int gcd(int a, int b) {
  while (b != 0)  {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__host__ int next_power_of_two(int x) {
    int pow = 1;
    while (pow < x) {
        pow *= 2;
    }
    return pow;
}

__device__ float get_squared_distance(float point1, float point2) {
    return (point1 - point2) * (point1 - point2);
}

__host__ int get_dimension(int total, int dim1){
    return (total+dim1-1)/dim1;
}

__host__ dim3 get_block_dimensions(int max_threads_per_block, int* max_thread_dimensions, int x, int y, int z){
    //to have 8 blocks per SM
    max_threads_per_block = max_threads_per_block/8;
    if(z!=1){
        z = gcd(max_threads_per_block, z<max_thread_dimensions[2] ? next_power_of_two(z>>1) : max_thread_dimensions[2]>>1);
    }
    if(y!=1){
        y = gcd(max_threads_per_block/z, y<max_thread_dimensions[1] ? next_power_of_two(y) : max_thread_dimensions[1]);
    }
    if(x!=1){
        x = gcd(max_threads_per_block/(z*y), x<max_thread_dimensions[0] ? next_power_of_two(x) : max_thread_dimensions[0]);
    }
    dim3 block(x,y,z);
    return block;
}

__host__ dim3 get_grid_dimensions(dim3 block, int x, int y, int z){
    dim3 grid(get_dimension(x, block.x),get_dimension(y, block.y),get_dimension(z, block.z));
    return grid;
}

__global__ void compute_distances(float *d_points, float *d_centroids, int dims, int n_vals, int n_clusters,
                                  float *d_point_centroid_distance) {
    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (point_id < n_vals && centroid_id < n_clusters && dim_id < dims) {
        atomicAdd(&d_point_centroid_distance[point_id * n_clusters + centroid_id],
                  get_squared_distance(d_points[point_id * dims + dim_id], d_centroids[centroid_id * dims + dim_id]));
    }
}

__global__ void compute_distances_shmem1(float *d_points, float *d_centroids, int dims, int n_vals, int n_clusters,
                                  float *d_point_centroid_distance) {

    extern __shared__ float points_smem[];
    
    float* centroids_smem = &points_smem[blockDim.x*blockDim.z];

    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (threadIdx.x==0 && centroid_id < n_clusters && dim_id < dims) {
        centroids_smem[threadIdx.y*blockDim.z+threadIdx.z] = d_centroids[centroid_id * dims + dim_id];
    }

    if (threadIdx.y==0 && point_id < n_vals && dim_id < dims) {
        points_smem[threadIdx.x*blockDim.z+threadIdx.z] = d_points[point_id * dims + dim_id];
    }

    __syncthreads();

    if (point_id < n_vals && centroid_id < n_clusters && dim_id < dims) {
        atomicAdd(&d_point_centroid_distance[point_id * n_clusters + centroid_id],
                  get_squared_distance(points_smem[threadIdx.x*blockDim.z+threadIdx.z], centroids_smem[threadIdx.y*blockDim.z+threadIdx.z]));
    }
}

__global__ void compute_distances_shmem2(float *d_points, float *d_centroids, int dims, int n_vals, int n_clusters,
                                  float *d_point_centroid_distance) {

    extern __shared__ float smem[];

    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (threadIdx.x==0 && centroid_id < n_clusters && dim_id < dims) {
        smem[threadIdx.y*blockDim.z+threadIdx.z] = d_centroids[centroid_id * dims + dim_id];
    }

    __syncthreads();

    if (point_id < n_vals && centroid_id < n_clusters && dim_id < dims) {
        atomicAdd(&d_point_centroid_distance[point_id * n_clusters + centroid_id],
                  get_squared_distance(d_points[point_id * dims + dim_id], smem[threadIdx.y*blockDim.z+threadIdx.z]));
    }
}

__global__ void compute_distances_shmem3(float *d_points, float *d_centroids, int dims, int n_vals, int n_clusters,
                                  float *d_point_centroid_distance) {

    extern __shared__ float smem[];

    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (threadIdx.y==0 && point_id < n_vals && dim_id < dims) {
        smem[threadIdx.x*blockDim.z+threadIdx.z] = d_points[point_id * dims + dim_id];
    }

    __syncthreads();

    if (point_id < n_vals && centroid_id < n_clusters && dim_id < dims) {
        atomicAdd(&d_point_centroid_distance[point_id * n_clusters + centroid_id],
                  get_squared_distance(smem[threadIdx.x*blockDim.z+threadIdx.z], d_centroids[centroid_id * dims + dim_id]));
    }
}

__global__ void compute_distances_shmem4(float *d_points, float *d_centroids, int dims, int n_vals, int n_clusters,
                                  float *d_point_centroid_distance) {

    extern __shared__ float smem[];

    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (point_id < n_vals && centroid_id < n_clusters && threadIdx.z==0) {
        smem[threadIdx.x*blockDim.y + threadIdx.y] = 0.0;
    }

    __syncthreads();

    if (point_id < n_vals && centroid_id < n_clusters && dim_id < dims) {
        atomicAdd(&smem[threadIdx.x*blockDim.y + threadIdx.y], get_squared_distance(d_points[point_id * dims + dim_id], d_centroids[centroid_id * dims + dim_id]));
    }

    __syncthreads();

    if (point_id < n_vals && centroid_id < n_clusters && threadIdx.z==0) {
        atomicAdd(&d_point_centroid_distance[point_id * n_clusters + centroid_id], smem[threadIdx.x*blockDim.y + threadIdx.y]);
    }
}

__global__ void assign_centroid_from_computed_distances(int *d_centroid_assignments, int n_vals, int n_clusters,
                                                        float *d_point_centroid_distance) {
    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (point_id < n_vals) {
        int distance_idx = point_id * n_clusters;

        int cluster_id = 0;
        float min_distance = d_point_centroid_distance[distance_idx];
        for (int i = distance_idx + 1; i < distance_idx + n_clusters; i++) {
            if (d_point_centroid_distance[i] < min_distance) {
                min_distance = d_point_centroid_distance[i];
                cluster_id = i - distance_idx;
            }
        }
        d_centroid_assignments[point_id] = cluster_id;
    }

}


__global__ void reset_distance_matrix(float *d_point_centroid_distance, int n_clusters, int n_vals, float *d_convergence_distances) {
    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    if(point_id<n_vals && centroid_id<n_clusters){
        d_point_centroid_distance[point_id * n_clusters + centroid_id] = 0.0;
        if(point_id==0){
            d_convergence_distances[centroid_id] = 0.0;
        }
    }
}

__global__ void reset_distance_matrix2(float *d_point_centroid_distance, int n_clusters, int n_vals) {
    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    if(point_id<n_vals && centroid_id<n_clusters){
        d_point_centroid_distance[point_id * n_clusters + centroid_id] = 0.0;
    }
}

__global__ void
swap_centroids(float *d_centroids, float *d_old_centroids, int *d_centroid_counts, int dims, int n_clusters) {
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;
    
    if(centroid_id<n_clusters && dim_id<dims){
        int centroid_idx = centroid_id * dims + dim_id;

        d_old_centroids[centroid_idx] = d_centroids[centroid_idx];
        d_centroids[centroid_idx] = 0.0;
        if (dim_id == 0) {
            d_centroid_counts[centroid_id] = 0;
        }
    }
}

__global__ void
reset_centroids(float *d_centroids, int *d_centroid_counts, int dims, int n_clusters) {
    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;
    
    if(centroid_id<n_clusters && dim_id<dims){
        int centroid_idx = centroid_id * dims + dim_id;
        d_centroids[centroid_idx] = 0.0;
        if (dim_id == 0) {
            d_centroid_counts[centroid_id] = 0;
        }
    }
}

__global__ void
accumulate_centroids(float *d_points, float *d_centroids, int *d_centroid_assignments, int dims, int n_vals,
                      int n_clusters, int *d_centroid_counts) {
    unsigned int point_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (point_id < n_vals && dim_id < dims){
        atomicAdd(&d_centroids[d_centroid_assignments[point_id] * dims + dim_id], d_points[point_id * dims + dim_id]);
        if (dim_id == 0) {
            atomicAdd(&d_centroid_counts[d_centroid_assignments[point_id]], 1);
        }
    }
}

__global__ void average_accumulated_centroids(float *d_centroids, int dims, int n_clusters, int *d_centroid_counts) {

    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (centroid_id < n_clusters && dim_id < dims){
        d_centroids[centroid_id * dims + dim_id] /= ((float) (d_centroid_counts[centroid_id]));
    }  
}

__global__ void calculate_convergence_distances(float *d_centroids, float *d_old_centroids, int dims, int n_clusters, float *d_convergence_distances) {

    unsigned int centroid_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int dim_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (centroid_id < n_clusters && dim_id < dims){
        atomicAdd(&d_convergence_distances[centroid_id], get_squared_distance(d_centroids[centroid_id * dims + dim_id], d_old_centroids[centroid_id * dims + dim_id]));
    }  
}

kmeans_iteration_t kmeans_cuda(cudaDeviceProp deviceProp, options_t opts, int n_vals, float *d_points, float *d_centroids,
                        int *d_centroid_assignments, int *d_centroid_counts, float *d_point_centroid_distances,
                        float **h_centroids, float **h_old_centroids) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int* max_thread_dimensions = deviceProp.maxThreadsDim;
    CHECK(cudaEventRecord(start));

    dim3 block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, n_vals, opts.num_cluster, 1);
    dim3 grid = get_grid_dimensions(block, n_vals, opts.num_cluster, 1);
    
    reset_distance_matrix2<<<grid,block>>>(d_point_centroid_distances, opts.num_cluster, n_vals);

    block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, n_vals, opts.num_cluster, opts.dims);
    grid = get_grid_dimensions(block, n_vals, opts.num_cluster, opts.dims);
    CHECK(cudaDeviceSynchronize());

    switch(opts.shared_memory) {
        case 1:
            compute_distances_shmem1<<<grid, block, (block.x*block.z + block.y*block.z)*sizeof(float)>>>(d_points, d_centroids, opts.dims, n_vals, opts.num_cluster, d_point_centroid_distances);
            break;
        case 2:
            compute_distances_shmem2<<<grid, block, block.y*block.z*sizeof(float)>>>(d_points, d_centroids, opts.dims, n_vals, opts.num_cluster, d_point_centroid_distances);
            break;
        case 3:
            compute_distances_shmem3<<<grid, block, block.x*block.z*sizeof(float)>>>(d_points, d_centroids, opts.dims, n_vals, opts.num_cluster, d_point_centroid_distances);
            break;
        case 4:
            compute_distances_shmem4<<<grid, block, block.x*block.y*sizeof(float)>>>(d_points, d_centroids, opts.dims, n_vals, opts.num_cluster, d_point_centroid_distances);
            break;
        default:
            compute_distances<<<grid, block>>>(d_points, d_centroids, opts.dims, n_vals, opts.num_cluster, d_point_centroid_distances);
    }

    block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, n_vals, 1, 1);
    grid = get_grid_dimensions(block, n_vals, 1, 1);
    CHECK(cudaDeviceSynchronize());

    assign_centroid_from_computed_distances<<<n_vals/1024, 1024>>>(d_centroid_assignments, n_vals, opts.num_cluster, d_point_centroid_distances);
    
    block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, 1, opts.num_cluster, opts.dims);
    grid = get_grid_dimensions(block, 1, opts.num_cluster, opts.dims);
    CHECK(cudaDeviceSynchronize());

    reset_centroids<<<grid, block>>>(d_centroids, d_centroid_counts, opts.dims, opts.num_cluster);

    block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, n_vals, 1, opts.dims);
    grid = get_grid_dimensions(block, n_vals, 1, opts.dims);
    CHECK(cudaDeviceSynchronize());
    
    accumulate_centroids<<<grid, block>>>(d_points, d_centroids, d_centroid_assignments, opts.dims, n_vals, opts.num_cluster, d_centroid_counts);

    block = get_block_dimensions(deviceProp.maxThreadsPerBlock, max_thread_dimensions, 1, opts.num_cluster, opts.dims);
    grid = get_grid_dimensions(block, 1, opts.num_cluster, opts.dims);
    CHECK(cudaDeviceSynchronize());

    average_accumulated_centroids<<<grid, block>>>(d_centroids, opts.dims, opts.num_cluster, d_centroid_counts);

    CHECK(cudaDeviceSynchronize());
    
    swap_centroids(h_old_centroids, h_centroids);
    cudaMemcpy(*h_centroids, d_centroids, opts.num_cluster*opts.dims*sizeof(float), cudaMemcpyDeviceToHost);
    kmeans_iteration_t iter;
    iter.converged = test_convergence(&opts, *h_old_centroids, *h_centroids);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&(iter.time_taken), start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return iter;
}