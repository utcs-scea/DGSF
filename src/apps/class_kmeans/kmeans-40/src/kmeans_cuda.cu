#include <chrono>
#include "io.h"
#include "kmeans.h"

#define BLOCK_SIZE 256

using namespace std;

static unsigned long int next_rand = 1;
static unsigned long kmeans_rmax = 32767;
void kmeans_srand(unsigned int seed) {
    next_rand = seed;
}
int kmeans_rand() {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand/65536) % (kmeans_rmax+1);
}


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


void init_centroids(struct options_t* args,
                    REAL*           dataset,
                    REAL*           centroids) {

    for (int k = 0; k < args->num_cluster; k++) {
        int index = kmeans_rand() % args->num_points;
        for (int d = 0; d < args->dims; d++) {
            centroids[k*args->dims+d] = dataset[index*args->dims+d];
        }
    }

}


__global__ void find_nearest_centroids_kernel(REAL* dataset,
                                              REAL* centroids,
                                              int*    labels,
                                              REAL* sum_centroids,
                                              int*    hit_counts,
                                              int     num_points,
                                              int     num_cluster,
                                              int     dims){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double nearest_distance = 1000000.0;
    int    nearest_centroid = 0;
    for (int k = 0; k < num_cluster; k++) {
        double accum_diff = 0.0;
        double dist;
        for (int d = 0; d < dims; d++) {
            accum_diff += (dataset[i*dims+d] - centroids[k*dims+d]) * (dataset[i*dims+d] - centroids[k*dims+d]);
        }
        dist = sqrt(accum_diff);
        if(dist < nearest_distance){
            nearest_distance = dist;
            nearest_centroid = k;
        }
    }
    if(i < num_points){
        labels[i] = nearest_centroid;
        atomicAdd(&hit_counts[labels[i]], 1);
        for (int d = 0; d < dims; d++) {
            //atomicAddDouble(&sum_centroids[labels[i]*dims+d], dataset[i*dims+d]);
            atomicAdd(&sum_centroids[labels[i]*dims+d], dataset[i*dims+d]);
        }
    }
}


__global__ void average_labeled_centroids_kernel(REAL* centroids,
                                                 REAL* sum_centroids,
                                                 int*    hit_counts,
                                                 int     num_cluster,
                                                 int     dims){
  
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < num_cluster){
        for (int d = 0; d < dims; d++) {
            centroids[i*dims+d] = sum_centroids[i*dims+d] / hit_counts[i];
        }
    }

}


__global__ void check_convergence(int*    labels,
                                  int*    old_labels,
                                  int*    is_converged,
                                  int     num_points){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < num_points){
        if(labels[i] != old_labels[i]){
            *is_converged = 0;
        }
    }
}


void kmeans_cuda(struct options_t* args,
                 REAL**            dataset,
                 int**             labels,
                 REAL**            centroids,
                 float*            time_loops, 
                 int*              iter_to_converge) {

    // for time measurements
    cudaEvent_t d_start, d_stop;
    cudaEvent_t d_start2, d_stop2;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventCreate(&d_start2);
    cudaEventCreate(&d_stop2);

    // initialize centroids randomly
    kmeans_srand(args->seed);
    init_centroids(args, *dataset, *centroids);

    // Alloc space for device copies 
    REAL* dataset_dev;                
    int*    labels_dev;                
    int*    old_labels_dev;                
    REAL* centroids_dev;
    REAL* sum_centroids_dev;
    int*    hit_counts_dev;
    int*    is_converged_dev;
    cudaMalloc((void**)&dataset_dev, args->num_points * sizeof(REAL) * args->dims);
	cudaMalloc((void**)&labels_dev, args->num_points * sizeof(int));
	cudaMalloc((void**)&old_labels_dev, args->num_points * sizeof(int));
	cudaMalloc((void**)&centroids_dev, args->num_cluster * sizeof(REAL) * args->dims);
	cudaMalloc((void**)&sum_centroids_dev, args->num_cluster * sizeof(REAL) * args->dims);
	cudaMalloc((void**)&hit_counts_dev, args->num_cluster * sizeof(int));
	cudaMalloc((void**)&is_converged_dev, sizeof(int));

    // Transfer to device
    cudaEventRecord(d_start);
    cudaMemcpy(dataset_dev, *dataset, args->num_points * sizeof(REAL) * args->dims, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_dev, *centroids, args->num_cluster * sizeof(REAL) * args->dims, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_dev, *labels, args->num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(d_stop);

    cudaEventSynchronize(d_stop);
    float milliseconds1 = 0.0;
    cudaEventElapsedTime(&milliseconds1, d_start, d_stop);

    // core algorithm
    int iterations = 0;
    int stable_count = 0;
    bool done = false;
    bool is_converged = false;
    auto start = std::chrono::high_resolution_clock::now();
    while(!done) {

        cudaMemcpy(old_labels_dev, labels_dev, args->num_points * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(hit_counts_dev, 0, args->num_cluster * sizeof(int));
        cudaMemset(sum_centroids_dev, 0, args->num_cluster * sizeof(REAL) * args->dims);
        cudaMemset(is_converged_dev, 1, sizeof(int));
        iterations++;

        // labels is a mapping from each point in the dataset 
        // to the nearest (euclidean distance) centroid
        find_nearest_centroids_kernel<<<(args->num_points+(BLOCK_SIZE-1))/BLOCK_SIZE,BLOCK_SIZE>>>(
                                               dataset_dev,
                                               centroids_dev,
                                               labels_dev,
                                               sum_centroids_dev,
                                               hit_counts_dev,
                                               args->num_points,
                                               args->num_cluster,
                                               args->dims);
 
        cudaDeviceSynchronize();

        // the new centroids are the average 
        // of all the points that map to each 
        // centroid
        //average_labeled_centroids_kernel<<<args->num_points/args->num_points,args->num_points>>>(
        average_labeled_centroids_kernel<<<(args->num_points+(BLOCK_SIZE-1))/BLOCK_SIZE,BLOCK_SIZE>>>(
                                               centroids_dev,
                                               sum_centroids_dev,
                                               hit_counts_dev,
                                               args->num_cluster,
                                               args->dims);
        //printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
        cudaDeviceSynchronize();

        //check_convergence<<<args->num_cluster/args->num_cluster,args->num_cluster>>>(
        check_convergence<<<(args->num_points+(BLOCK_SIZE-1))/BLOCK_SIZE,BLOCK_SIZE>>>(
                                  labels_dev,
                                  old_labels_dev,
                                  is_converged_dev,
                                  args->num_points);
        cudaDeviceSynchronize();
        cudaMemcpy(&is_converged, is_converged_dev, sizeof(int), cudaMemcpyDeviceToHost);
        if(is_converged==1){
            stable_count++;
        }else{
            stable_count = 0;
        }
        done = iterations > args->max_num_iter || stable_count==5;

    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    *time_loops = diff.count()/1000.0;
    *iter_to_converge = iterations;


    // Copy result back to host
    cudaEventRecord(d_start2);
    cudaMemcpy(*labels, labels_dev, args->num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*centroids, centroids_dev, args->num_cluster * sizeof(REAL) * args->dims, cudaMemcpyDeviceToHost);
    cudaEventRecord(d_stop2);

    // Cleanup
    cudaFree(dataset_dev);
    cudaFree(labels_dev);
    cudaFree(old_labels_dev);
    cudaFree(centroids_dev);
    cudaFree(sum_centroids_dev);
    cudaFree(hit_counts_dev);
    cudaFree(is_converged_dev);
    
    cudaEventSynchronize(d_stop2);
    float milliseconds2 = 0.0;
    cudaEventElapsedTime(&milliseconds2, d_start2, d_stop2);
    if(args->end2end){
        cout << "data transfer: " << milliseconds1 + milliseconds2 << endl;
    }
    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
    cudaEventDestroy(d_start2);
    cudaEventDestroy(d_stop2);

    return;

} 

