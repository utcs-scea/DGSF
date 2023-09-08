#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_BLOCKS 65536
#define MAX_THREADS 256



// the below function has been taken from online sources for atomicAdd on doubles
__device__ double my_atomicAdd(double* address, double val) {
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


// random centroid generation code
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

// function to copy centroids to old centroids
__global__ void copy_old_centroids(double **old_centroids, double **centroids, 
                                   int num_clusters, int dims) {
    int totalthreads = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < num_clusters) {
        for (int i=0; i<dims; i++) {
            old_centroids[index][i] = centroids[index][i];
        }
        index += totalthreads;
    }
}

__global__ void find_nearest_centroids(double **points, double **centroids, 
                                  int num_inputs, int num_clusters, int dims, 
                                  int *labels) {
    int totalthreads = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double shared_centroid_coords[];
    int threadId = threadIdx.x;
    while (threadId < num_clusters) {
        for (int i=0; i<dims; i++) {
            shared_centroid_coords[threadId*dims + i] = centroids[threadId][i];
        }
        threadId += blockDim.x;
    }
    __syncthreads();
    
    
    while (index < num_inputs) {
        double min_distance = 0.0;
        double min_centroid = -1;
        for (int i=0; i<num_clusters; i++) {
            double distance = 0.0;
            int point_coord_pos = 0;
            for (int j=0; j<dims; j++) {
                distance += (shared_centroid_coords[i*dims+j]-points[index][point_coord_pos])
                    *(shared_centroid_coords[i*dims+j]-points[index][point_coord_pos]);
                point_coord_pos += num_inputs;
            }
            if (distance < min_distance || min_centroid < 0) {
                min_distance = distance;
                min_centroid = i;
            }
        }
        labels[index] = min_centroid;
        index += totalthreads;
    }  
}

__global__ void set_centroids_zero(double **centroids, int num_clusters, 
                                   int dims, int *label_count) {
    int totalthreads = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < num_clusters) {
        for (int i=0; i<dims; i++) {
            centroids[index][i] = 0.0;
        }
        label_count[index] = 0;
        index += totalthreads;
    }
}

__global__ void find_point_sum(double **points, double **centroids, int *labels, 
                               int num_inputs, int num_clusters, int dims, 
                               int *label_count) {
    int totalthreads = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double shared_centroid_coords[];
    // steal memory from double array which we allocated too large to 
    // form shared label counts
    int *shared_label_count = (int*)&(shared_centroid_coords[num_clusters*dims]);
    
    
    // initialize the shared data
    int threadId = threadIdx.x;
    while (threadId < num_clusters) {
        for (int i=0; i<dims; i++) {
            shared_centroid_coords[threadId*dims + (i+threadId)%dims] = 0.0;
        }
        shared_label_count[threadId] = 0;
        threadId += blockDim.x;
    }
    
    // fill the shared data using atomic add
    __syncthreads();
    while (index < num_inputs) {
        int point_coord_pos = 0;
        for (int i=0; i<dims; i++) {
            my_atomicAdd(&(shared_centroid_coords[labels[index]*dims + i]), 
                      points[index][point_coord_pos]);
            point_coord_pos += num_inputs;
        }
        atomicAdd(&(shared_label_count[labels[index]]), 1);
        index += totalthreads;
    }
    
    
    // update the main memory using the shared data
    __syncthreads();
    threadId = threadIdx.x;
    while (threadId < num_clusters) {
        for (int i=0; i<dims; i++) {
            my_atomicAdd(&(centroids[threadId][(i+threadId)%dims]), 
                      shared_centroid_coords[threadId*dims + (i+threadId)%dims]);
        }
        atomicAdd(&(label_count[threadId]), shared_label_count[threadId]);
        threadId += blockDim.x;
    }
    
}

__global__ void find_centroids(double **centroids, int num_clusters, int dims, 
                               int *label_count) {
    int totalthreads = blockDim.x*gridDim.x;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    while (index < num_clusters) {
        for (int i=0; i<dims; i++) {
            centroids[index][i]/=double(label_count[index]);
        }
        index += totalthreads;
    }
}

__global__ void find_point_difference(double **old_centroids, double **centroids, 
                                      int num_clusters, int dims, double *result, 
                                      double threshold) {
    __shared__ double block_distance[1];
    block_distance[0] = 0.0;
    __syncthreads();
    int totalthreads = blockDim.x*gridDim.x;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    while (index < num_clusters) {
        double distance = 0.0;
        for (int i=0; i<dims; i++) {
            distance += (old_centroids[index][i] - centroids[index][i])
                *(old_centroids[index][i] - centroids[index][i]);
        }
        my_atomicAdd(block_distance, distance);
        index += totalthreads;
    }
    if (threadIdx.x == 0) {
        my_atomicAdd(result, block_distance[0]);
    }
}


int main(int argc, char* argv[]) {
    // take the commandline options
    int num_clusters, dims, max_num_iters, seed;
    char* fileName;
    double threshold;
    bool outputCentroids = false;
    int c;
    while ((c = getopt(argc, argv, "k:d:i:m:t:s:c")) != -1) {
        switch(c) {
            case 'k':
                num_clusters = atoi((char*)optarg);
                break;
            case 'd':
                dims = atoi((char*)optarg);
                break;
            case 'i':
                fileName = (char*)optarg;
                break;
            case 'm':
                max_num_iters = atoi((char*)optarg);
                break;
            case 't':
                threshold = atof((char*)optarg)/double(10.0);
                threshold = threshold*threshold;
                break;
            case 's':
                seed = atoi((char*)optarg);
                break;
            case 'c':
                outputCentroids = true;
                break;
        }
    }
    
    FILE *fp = fopen(fileName, "r");
    int num_inputs;
    c = fscanf(fp, "%d", &num_inputs);
    std::vector<double*> host_points(num_inputs);
    
    double **device_points_data = (double**)malloc(num_inputs*sizeof(double*));
    
    // read input from file and copy to device memory
    int input_pos;
    double *tmp_data = (double*)malloc(num_inputs*dims*sizeof(double));
    for (int i=0; i<num_inputs; i++) {
//         host_points[i] = (double*) malloc(dims*sizeof(double));
        host_points[i] = tmp_data+(i);
        c = fscanf(fp, "%d", &input_pos);
        for (int j=0; j<dims*num_inputs; j+=num_inputs) {
            c = fscanf(fp, "%lf", &(host_points[i][j]));
        }
        
    }
    
    
    // prepare the centroids
    std::vector<double*> host_centroids(num_clusters);
    
    double **device_centroids_data = (double**)malloc(num_clusters*sizeof(double*));
    double **old_host_centroids = (double**)malloc(num_clusters*sizeof(double*));
    
    
    // fill the centroids
    kmeans_srand(seed);
    tmp_data = (double*)malloc(num_clusters*dims*sizeof(double));
    for (int i=0; i<num_clusters; i++) {
        // compute the current centroids using the randomization
//         host_centroids[i] = (double*)malloc(dims*sizeof(double));
        host_centroids[i] = tmp_data + (i*dims);
        int index = kmeans_rand() % num_inputs;
        int point_coord_pos = 0;
        for (int j=0; j<dims; j++) {
            host_centroids[i][j] = host_points[index][point_coord_pos];
            point_coord_pos += num_inputs;
        }
    }
    
    // club all cuda memcpy and cuda malloc together
    double **device_points;
    cudaMalloc((void**)&device_points, num_inputs*sizeof(double*));
    double **device_centroids;
    cudaMalloc((void**)&device_centroids, num_clusters*(sizeof(double*)));
    double **old_device_centroids;
    cudaMalloc((void**)&old_device_centroids, num_clusters*sizeof(double*));
    
    cudaMalloc((void**)&tmp_data, num_inputs*dims*sizeof(double));
    for (int i=0; i<num_inputs; i++) {
        device_points_data[i] = tmp_data+(i);
//         cudaMalloc((void**)&(device_points_data[i]), dims*sizeof(double));
//         cudaMemcpy(device_points_data[i], host_points[i], 
//                    dims*sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(device_points_data[0], host_points[0], 
               num_inputs*dims*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_points, device_points_data, 
               num_inputs*sizeof(double*), cudaMemcpyHostToDevice);
    
//     cudaMalloc((void**)&tmp_data, num_clusters*dims*sizeof(double));
//     double *tmp_data2;
//     cudaMalloc((void**)&tmp_data2, num_clusters*dims*sizeof(double));    
    
    for (int i=0; i<num_clusters; i++) {
//         device_centroids_data[i] = tmp_data + (i*dims);        
        cudaMalloc((void**)&(device_centroids_data[i]), dims*sizeof(double));
        cudaMemcpy(device_centroids_data[i], host_centroids[i], 
                   dims*sizeof(double), cudaMemcpyHostToDevice);
        
        // prepare the storage space for old centroids
//         old_host_centroids[i] = tmp_data2 + (i*dims);
        cudaMalloc((void**)&(old_host_centroids[i]), dims*sizeof(double));
    }
//     cudaMemcpy(device_centroids_data[0], host_centroids[0], 
//                num_clusters*dims*sizeof(double), cudaMemcpyHostToDevice);    
    cudaMemcpy(device_centroids, device_centroids_data, 
               num_clusters*sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(old_device_centroids, old_host_centroids, 
               num_clusters*sizeof(double*), cudaMemcpyHostToDevice);
    
    // find labels using kernel
    int *host_labels = (int*)malloc(num_inputs*sizeof(int));
    int *device_labels;
    cudaMalloc((void**)&device_labels, num_inputs*sizeof(int));
    int *label_count;
    cudaMalloc((void**)&label_count, num_clusters*sizeof(int));
    
    // space for convergence scores
    double *host_convergence_score = (double*)malloc(sizeof(double));
    double *device_convergence_score;
    cudaMalloc((void**)&device_convergence_score, sizeof(double));
    
    // cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int iters = 0;
    int shouldBreak = 0;
    int threads_per_block = MAX_THREADS;
    int blocks = 0;
    while (iters < max_num_iters) {
        // copy current centroids to old centroids
        // copy the current centroids to old centroids
        threads_per_block = 16;
        blocks = min(MAX_BLOCKS, (num_clusters+threads_per_block-1)/threads_per_block);
        copy_old_centroids<<<blocks, threads_per_block>>>(old_device_centroids, 
                                                          device_centroids,
                                                         num_clusters, dims);
        
        // find the nearest centroid. Use shmem in doing so.
        threads_per_block = MAX_THREADS;
        blocks = min(MAX_BLOCKS, (num_inputs+threads_per_block-1)/threads_per_block);
        find_nearest_centroids<<<blocks, threads_per_block, num_clusters*dims*sizeof(double)>>>(device_points,
                                                              device_centroids,
                                                              num_inputs, num_clusters, 
                                                              dims, device_labels);

        
        // calculate the sum and number of points assigned to each cluster
        threads_per_block = 16;
        blocks = min(MAX_BLOCKS, (num_clusters+threads_per_block-1)/threads_per_block);
        set_centroids_zero<<<blocks, threads_per_block>>>(device_centroids, num_clusters, 
                                                          dims, label_count);
        
        
        threads_per_block = MAX_THREADS;
        blocks = min(MAX_BLOCKS, (num_inputs+threads_per_block-1)/threads_per_block);
        find_point_sum<<<blocks, threads_per_block, 
            num_clusters*dims*sizeof(double) + num_clusters*sizeof(int)>>>(device_points, 
                                                                    device_centroids, 
                                                                    device_labels, 
                                                                    num_inputs, 
                                                                    num_clusters, 
                                                                    dims, label_count);
        
        // calculate the centroids of each cluster
        threads_per_block = 16;
        blocks = min(MAX_BLOCKS, (num_clusters+threads_per_block-1)/threads_per_block);
        find_centroids<<<blocks, threads_per_block>>>(device_centroids, num_clusters, 
                                                      dims, label_count);
        iters++;
        
        // calculate convergence score and break if it is below threshold
        (*host_convergence_score) = 0.0;
        cudaMemcpy(device_convergence_score, host_convergence_score, 
                   sizeof(double), cudaMemcpyHostToDevice);
        threads_per_block = 16;
        blocks = min(MAX_BLOCKS, (num_clusters+threads_per_block-1)/threads_per_block);
        find_point_difference<<<blocks, threads_per_block>>>(old_device_centroids, 
                                                             device_centroids,
                                                             num_clusters, dims, 
                                                             device_convergence_score,
                                                            threshold);
        cudaMemcpy(host_convergence_score, device_convergence_score, 
                   sizeof(double), cudaMemcpyDeviceToHost);
        if ((*host_convergence_score) < threshold) {
            shouldBreak += 1;
        } else {
            shouldBreak = 0;
        }
        if (shouldBreak>3) {
            break;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%d,%lf\n", iters, time/double(iters));
    
    
    
    // print the required thing here
    if (outputCentroids) {
        // copy the centroids from device to host
        for (int i=0; i<num_clusters; i++) {
            cudaMemcpy(host_centroids[i], device_centroids_data[i], dims*sizeof(double), 
                       cudaMemcpyDeviceToHost);
        }
        // print the centroids
        for (int i=0; i<num_clusters; i++) {
            printf("%d ", i);
            for (int j=0; j<dims; j++) {
                printf("%lf ", host_centroids[i][j]);
            }
            printf("\n");
        }
    } else {
        // copy labels from device to host
        cudaMemcpy(host_labels, device_labels, num_inputs*sizeof(int), cudaMemcpyDeviceToHost);
        // print the labels
        printf("clusters:");
        for (int p=0; p < num_inputs; p++)
            printf(" %d", host_labels[p]);
    }
    
    // free memory here
    // well don't free the memory to save time
    
//     for (int i=0; i<num_inputs; i++) {
//         free(host_points[i]);
//         cudaFree(device_points_data[i]);
//     }
//     free(device_points_data);
//     cudaFree(device_points);
    
//     for (int i=0; i<num_clusters; i++) {
//         free(host_centroids[i]);
//         cudaFree(device_centroids_data[i]);
//         cudaFree(old_host_centroids[i]);
//     }
//     free(device_centroids_data);
//     cudaFree(device_centroids);
//     free(old_host_centroids);
//     cudaFree(old_device_centroids);
//     free(host_convergence_score);
//     cudaFree(device_convergence_score);
//     free(host_labels);
//     cudaFree(device_labels);
//     cudaFree(label_count);
    
}