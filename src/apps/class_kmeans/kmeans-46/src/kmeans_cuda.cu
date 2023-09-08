#include "kmeans_cuda.h"

#define SQ(v) ((v) * (v))
#define MAX_SHARED_MEM 48 * 1024

/*
 * Common CUDA Functions 
 */

__device__ float cuda_calc_sq_dist(int     num_dims, 
                                   float* i1, 
                                   float* i2) {
    float sq_dist = 0; 

    for (int d = 0; d < num_dims; d++) {
        sq_dist += SQ(i1[d] - i2[d]);
    }
    return sq_dist;
}

__device__ int cuda_find_min_dist_centroid(struct options_t* d_opts,
                                           int               data_idx, 
                                           float*            data,
                                           float*            d_curr_centroids) {
    float min_sq_dist = FLT_MAX;
    int min_cluster    = 0;
    for (int i = 0; i < d_opts->k_clusters; i++) {
        float* data_pt        = data + (data_idx * d_opts->n_dims);
        float* centroid_pt    = d_curr_centroids + (i * d_opts->n_dims);
        float cluster_sq_dist = cuda_calc_sq_dist(d_opts->n_dims, data_pt, centroid_pt);
     
        if (cluster_sq_dist < min_sq_dist) { 
            min_sq_dist = cluster_sq_dist; 
            min_cluster = i;
        }
    }
    return min_cluster;
}

__global__ void cuda_check_convergence(bool*             done,
                                       struct options_t* d_opts,
                                       float*            d_old_centroids,
                                       float*            d_new_centroids) { 
    for (int i = 0; i < d_opts->k_clusters; i++) {
        float* old_centroid_pt = d_old_centroids + (d_opts->n_dims * i);
        float* new_centroid_pt = d_new_centroids + (d_opts->n_dims * i);
        float sq_dist          = cuda_calc_sq_dist(d_opts->n_dims, old_centroid_pt, new_centroid_pt);
        if (sq_dist > d_opts->threshold) {
            *done = false;
            return;
        }
    }
    *done = true;
}

/* Taken from https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/ */
void
printCudaInfo() {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Memory Clock Rate (KHz): %d\n",
           deviceProps.memoryClockRate);
        printf("   Memory Bus Width (bits): %d\n",
               deviceProps.memoryBusWidth);
        printf("   Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*deviceProps.memoryClockRate*(deviceProps.memoryBusWidth/8)/1.0e6);
        
    }
    printf("---------------------------------------------------------\n");
}

/*
 * Cuda Basic Functions 
 */

__global__ void cuda_add_point_to_centroid(float*            d_new_centroid_sums, 
                                           int*              d_new_centroid_cnts,
                                           struct options_t* d_opts, 
                                           float*            d_data, 
                                           int               n_vals,
                                           float*            d_curr_centroids) {
    
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (data_idx >= n_vals) {
        return;
    }
    
    int min_cluster = cuda_find_min_dist_centroid(d_opts, data_idx, d_data, d_curr_centroids);
    for (int i = 0; i < d_opts->n_dims; i++) {
        float* centroid_pt = d_new_centroid_sums + (min_cluster * d_opts->n_dims) + i;
        float pt_data     = d_data[(data_idx * d_opts->n_dims) + i];
        atomicAdd(centroid_pt, pt_data);
    }
    atomicAdd(&d_new_centroid_cnts[min_cluster], 1);
}

__global__ void cuda_avg_centroid_points(float*            d_centroids,
                                         struct options_t* d_opts,
                                         int*              d_centroid_cnts) {
                                    
    int cluster_idx = threadIdx.x;
    for (int i = 0; i < d_opts->n_dims; i++) {
        int centroid_idx = (cluster_idx * d_opts->n_dims) + i;
        d_centroids[centroid_idx] /= d_centroid_cnts[cluster_idx];
    }
}

/* 
 * Cuda Shared Memory Functions 
 */

__global__ void cuda_add_point_to_centroid_per_block(float*            d_new_centroid_sums, 
                                                     int*              d_new_centroid_cnts,
                                                     struct options_t* d_opts, 
                                                     float*            d_data, 
                                                     int               n_vals,
                                                     float*            d_curr_centroids) {
    extern __shared__ float shmem_data[];
    
    // 'local_idx' is used to index into shared memory 
    // (reserved 'n_dims+1' float's for each point in a block)
    int local_idx = threadIdx.x;
    
    // 'data_idx' is used to index into global memory (data-points)
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // zero out all shared memory (counts/points)
    for (int i = 0; i < d_opts->n_dims; i++) {
        shmem_data[(local_idx * d_opts->n_dims) + i] = 0;
    }
    shmem_data[(blockDim.x * d_opts->n_dims) + local_idx] = 0;
    __syncthreads();

    // load all centroids into shared memory
    if (local_idx < d_opts->k_clusters) {
        for (int i = 0; i < d_opts->n_dims; i++) {
            int cluster_dim_idx = (local_idx * d_opts->n_dims) + i;
            shmem_data[cluster_dim_idx] = d_curr_centroids[cluster_dim_idx];
        }
    }  
    
    int min_cluster = -1;
    if (data_idx < n_vals) {
        // find the closest centroid for the given point (all centroids are currently stored in shared memory)
        min_cluster = cuda_find_min_dist_centroid(d_opts, data_idx, d_data, shmem_data);
    }

    // before overwriting centroid data by the points that belong to each centroid batch, we need to ensure
    // that all threads have found the closest centroid to the given data-point
    __syncthreads();
   
    // zero out the centroids
    if (local_idx < d_opts->k_clusters) {
        for (int i = 0; i < d_opts->n_dims; i++) {
            shmem_data[(local_idx * d_opts->n_dims) + i] = 0;
        }
    }
     __syncthreads();
     
     if (local_idx >= n_vals) {
         return;
     }
    
    // for each centroid, find all points handled by the current block that map to it (ie 'min_cluster == i'); 
    // set the 'dim' values for the 'local_idx' point to '0' if the 'local_idx' point isn't in the cluster 
    // being currently processed    
    for (int i = 0; i < d_opts->k_clusters; i++) {
        int pt_in_cluster = (min_cluster == i);
        for (int j = 0; j < d_opts->n_dims; j++) {
            int shmem_dim_idx = (local_idx * d_opts->n_dims) + j;
            int data_dim_idx  = (data_idx * d_opts->n_dims) + j;
            shmem_data[shmem_dim_idx] = pt_in_cluster ? d_data[data_dim_idx] : 0;
        }
        int count_idx = (blockDim.x * d_opts->n_dims) + local_idx;
        shmem_data[count_idx] = pt_in_cluster;
              
        __syncthreads();
        
        // reduce over all the values in shared memory in parallel (similar to prefix-sum)
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (local_idx < stride) {
                int shmem_dim_idx_base        = local_idx * d_opts->n_dims;
                int shmem_dim_idx_stride_base = (local_idx + stride) * d_opts->n_dims;
                for (int j = 0; j < d_opts->n_dims; j++) {
                    if (shmem_dim_idx_stride_base < d_opts->n_dims * blockDim.x) {
                        shmem_data[shmem_dim_idx_base + j] += shmem_data[shmem_dim_idx_stride_base + j];
                    }
                } 
                int count_idx = (blockDim.x * d_opts->n_dims) + local_idx;
                if (count_idx + stride < (d_opts->n_dims + 1) * blockDim.x) {
                    shmem_data[count_idx] += shmem_data[count_idx + stride];
                }
            }
            // synchronize on each prefix-sum phase
            __syncthreads();
        }
        __syncthreads();
        
        // results are stored at the 0th point in shared memory; write the results back to the global arrays
        if (local_idx == 0) {
            for (int j = 0; j < d_opts->n_dims; j++) {
                int centroid_dim_sum_idx = (blockIdx.x * d_opts->k_clusters * d_opts->n_dims) + (i * d_opts->n_dims) + j;
                d_new_centroid_sums[centroid_dim_sum_idx] = shmem_data[j];
            }
            int centroid_cnt_idx = (blockIdx.x * d_opts->k_clusters + i);
            d_new_centroid_cnts[centroid_cnt_idx] = (int)shmem_data[blockDim.x * d_opts->n_dims];
        }
        // don't process next batch of centroids until threads have finished processing this batch
        __syncthreads();
    }
}

__global__ void cuda_avg_centroid_points_all_blocks(float*            d_centroids,
													struct options_t* d_opts,
													int               num_blocks,
													float*            d_new_centroid_sums,
													int*              d_centroid_cnts) {
                                    
    int cluster_idx = threadIdx.x;
    
	int num_centroid_pts = 0;
	for (int i = 0; i < num_blocks; i++) {
		for (int j = 0; j < d_opts->n_dims; j++) {
            int centroid_idx = (cluster_idx * d_opts->n_dims) + j;
            int centroid_sum_idx = (i * d_opts->k_clusters * d_opts->n_dims) + centroid_idx;
			d_centroids[centroid_idx] += d_new_centroid_sums[centroid_sum_idx];
		}
		num_centroid_pts += d_centroid_cnts[(i * d_opts->k_clusters) + cluster_idx];
    }

	for (int i = 0; i < d_opts->n_dims; i++) {
        int centroid_idx = (cluster_idx * d_opts->n_dims) + i;
        d_centroids[centroid_idx] /= num_centroid_pts;
	}
}

/* 
 * Cuda Implementations 
 */

void kmeans_cuda_basic(float**           centroids_p, 
					   int*              iterations_p, 
					   int*              copy_milliseconds_p,
					   int*              exec_milliseconds_p, 
					   struct options_t* h_opts, 
					   float*            h_input_vals, 
					   int               n_vals) {  
	int num_input_vals_bytes   = n_vals * h_opts->n_dims * sizeof(float);
	int num_centroids_bytes    = h_opts->k_clusters * h_opts->n_dims * sizeof(float);
	int num_centroid_cnt_bytes = h_opts->k_clusters * sizeof(int);

    int threads_per_block = 512;
	int num_blocks        = (n_vals + threads_per_block - 1) / threads_per_block;

    auto copy_start = std::chrono::high_resolution_clock::now(); 

	float* d_input_vals;
	cudaMalloc(&d_input_vals, num_input_vals_bytes);
	cudaMemcpy(d_input_vals, h_input_vals, num_input_vals_bytes, cudaMemcpyHostToDevice);

	struct options_t* d_opts;
	cudaMalloc(&d_opts, sizeof(struct options_t));
	cudaMemcpy(d_opts, h_opts, sizeof(struct options_t), cudaMemcpyHostToDevice);

	float* h_curr_centroids = *centroids_p;
	float* d_curr_centroids;
	cudaMalloc(&d_curr_centroids, num_centroids_bytes);
	cudaMemcpy(d_curr_centroids, h_curr_centroids, num_centroids_bytes, cudaMemcpyHostToDevice);
	
	float* d_new_centroids;
	cudaMalloc(&d_new_centroids,   num_centroids_bytes);
	cudaMemset(d_new_centroids, 0, num_centroids_bytes);
	
	int*   d_new_centroid_cnts;
	cudaMalloc(&d_new_centroid_cnts,   num_centroid_cnt_bytes);
	cudaMemset(d_new_centroid_cnts, 0, num_centroid_cnt_bytes);

    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
    int copy_milliseconds = copy_diff.count(); 

    int exec_milliseconds = 0;
    int iterations   = 0;
    
    bool  h_done  = false;
    bool* d_done;
    cudaMalloc(&d_done, sizeof(bool));
    cudaMemset(d_done, 0, sizeof(bool));

	while (!h_done) {
        auto exec_start = std::chrono::high_resolution_clock::now(); 
		cuda_add_point_to_centroid<<<num_blocks, threads_per_block>>>(d_new_centroids,
                                                                   d_new_centroid_cnts,
                                                                   d_opts,
                                                                   d_input_vals,
                                                                   n_vals,
                                                                   d_curr_centroids);
		cudaDeviceSynchronize();
		        
		cuda_avg_centroid_points<<<1, h_opts->k_clusters>>>(d_new_centroids,
                                                            d_opts, 
                                                            d_new_centroid_cnts);
        cudaDeviceSynchronize();
		
		iterations++;
        if (iterations >= h_opts->max_num_iters) {
            h_done = true;
        } else {
			cuda_check_convergence<<<1,1>>>(d_done, 
                                            d_opts, 
                                            d_curr_centroids, 
                                            d_new_centroids);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
                
	    float* d_temp_centroids = d_curr_centroids;
        d_curr_centroids = d_new_centroids;
        d_new_centroids  = d_temp_centroids;
        
        cudaMemset(d_new_centroids,     0, num_centroids_bytes);
        cudaMemset(d_new_centroid_cnts, 0, num_centroid_cnt_bytes);

        auto exec_end = std::chrono::high_resolution_clock::now();
		auto exec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_start);   
        exec_milliseconds += exec_diff.count(); 
	}
	
    copy_start = std::chrono::high_resolution_clock::now(); 
	
    cudaMemcpy(h_curr_centroids, d_curr_centroids, num_centroids_bytes, cudaMemcpyDeviceToHost);
	
    copy_end = std::chrono::high_resolution_clock::now();
    copy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
    copy_milliseconds += copy_diff.count(); 
  
    *copy_milliseconds_p = copy_milliseconds;
    *exec_milliseconds_p = exec_milliseconds;
    *iterations_p   = iterations;
}	  

void kmeans_cuda_shmem(float**           centroids_p, 
					   int*              iterations_p, 
					   int*              copy_milliseconds_p,
					   int*              exec_milliseconds_p, 
					   struct options_t* h_opts, 
					   float*            h_input_vals, 
					   int               n_vals)  {
    int num_input_vals_bytes   = n_vals * h_opts->n_dims * sizeof(float);
	int num_centroids_bytes    = h_opts->k_clusters * h_opts->n_dims * sizeof(float);
	int num_centroid_cnt_bytes = h_opts->k_clusters * sizeof(int);

    int threads_per_block = 256;
	int num_blocks        = (n_vals + threads_per_block - 1) / threads_per_block;
    // <x0, y0, z0>, <x1, y1, z1>, ... <xn, yn, zn>, <cnt0, cnt1, ..., cntn> (n = threads_per_block = no. points per block)
    int num_shmem_bytes   = (h_opts->n_dims + 1) * threads_per_block * sizeof(float);

    auto copy_start = std::chrono::high_resolution_clock::now(); 

	float* d_input_vals;
	cudaMalloc(&d_input_vals, num_input_vals_bytes);
    cudaMemcpy(d_input_vals, h_input_vals, num_input_vals_bytes, cudaMemcpyHostToDevice);


	struct options_t* d_opts;
	cudaMalloc(&d_opts, sizeof(struct options_t));
	cudaMemcpy(d_opts, h_opts, sizeof(struct options_t), cudaMemcpyHostToDevice);

	float* h_curr_centroids = *centroids_p;
	float* d_curr_centroids;
	cudaMalloc(&d_curr_centroids, num_centroids_bytes);
	cudaMemcpy(d_curr_centroids, h_curr_centroids, num_centroids_bytes, cudaMemcpyHostToDevice);
	
    // allocate 'num_blocks' sets of centroid sum values and centroid point counts; this allows 
    // us to compute the centroid sum/counts for each block independently using shared memory 
    // and then reduce the values later
    float* d_new_centroid_sums;
    cudaMalloc(&d_new_centroid_sums, num_centroids_bytes * num_blocks);    
    cudaMemset(d_new_centroid_sums, 0, num_centroids_bytes * num_blocks);
    
	int*   d_new_centroid_cnts;
	cudaMalloc(&d_new_centroid_cnts,   num_centroid_cnt_bytes * num_blocks);  
    cudaMemset(d_new_centroid_cnts, 0, num_centroid_cnt_bytes);
    
    float* d_new_centroids;
	cudaMalloc(&d_new_centroids,   num_centroids_bytes);   
    cudaMemset(d_new_centroids, 0, num_centroids_bytes);

    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
    int copy_milliseconds = copy_diff.count(); 

    int exec_milliseconds = 0;
    int iterations   = 0;
    
    bool  h_done  = false;
    bool* d_done;
    cudaMalloc(&d_done, sizeof(bool));
    cudaMemset(d_done, 0, sizeof(bool));

	while (!h_done) {
        auto exec_start = std::chrono::high_resolution_clock::now(); 

		cuda_add_point_to_centroid_per_block<<<num_blocks, threads_per_block, num_shmem_bytes>>>(d_new_centroid_sums,
                                                                                        d_new_centroid_cnts,
                                                                                        d_opts,
                                                                                        d_input_vals,
                                                                                        n_vals,
                                                                                        d_curr_centroids);  

        cudaDeviceSynchronize();

		cuda_avg_centroid_points_all_blocks<<<1, h_opts->k_clusters>>>(d_new_centroids,
																       d_opts,
																	   num_blocks,
																	   d_new_centroid_sums,
																	   d_new_centroid_cnts);

        cudaDeviceSynchronize();
		
		iterations++;
        if (iterations >= h_opts->max_num_iters) {
            h_done = true;
        } else {
			cuda_check_convergence<<<1,1>>>(d_done, 
                                            d_opts, 
                                            d_curr_centroids, 
                                            d_new_centroids);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
		}
                
	    float* d_temp_centroids = d_curr_centroids;
        d_curr_centroids = d_new_centroids;
        d_new_centroids  = d_temp_centroids;
        
        cudaMemset(d_new_centroids,     0, num_centroids_bytes);
        cudaMemset(d_new_centroid_cnts, 0, num_centroid_cnt_bytes);

        auto exec_end = std::chrono::high_resolution_clock::now();
		auto exec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_start);   
        exec_milliseconds += exec_diff.count(); 
	}
	
    copy_start = std::chrono::high_resolution_clock::now(); 
	
    cudaMemcpy(h_curr_centroids, d_curr_centroids, num_centroids_bytes, cudaMemcpyDeviceToHost);
	
    copy_end = std::chrono::high_resolution_clock::now();
    copy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
    copy_milliseconds += copy_diff.count(); 
  
    *copy_milliseconds_p = copy_milliseconds;
    *exec_milliseconds_p = exec_milliseconds;
    *iterations_p   = iterations;
}