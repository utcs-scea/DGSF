#include <kmeans.h>
#include <cfloat>

// The code is copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void calculate_distance_of_dims(double* distance_of_dims,
                                           double* device_input_vals,
                                           double* current_centroids,
                                           int n_vals,
                                           int num_cluster,
                                           int dims){
    int n_vals_index = blockIdx.x / num_cluster;
    int num_cluster_index = blockIdx.x % num_cluster;
    double x = device_input_vals[n_vals_index*dims + threadIdx.x];
    double y = current_centroids[num_cluster_index*dims + threadIdx.x];
    atomicAdd(distance_of_dims+n_vals_index*num_cluster+num_cluster_index,(x - y) * (x - y));
}

__global__ void get_nearest(int* device_clusterId_of_point,
                            double* distance_of_dims,
                            int n_vals,
                            int num_cluster,
                            int dims) {
      double min_distance =  DBL_MAX;
      device_clusterId_of_point[blockIdx.x]= -1;
      for(int i = 0; i < num_cluster; ++i) {
          double current_distance = distance_of_dims[blockIdx.x*num_cluster + i];
          if(current_distance < min_distance) {
              min_distance = current_distance;
              device_clusterId_of_point[blockIdx.x] = i;
          }
      }
 }


__global__ void add_to_centroids(int* device_clusterId_of_point,
                               double* device_input_vals,
                               double* current_centroids,
                               int* clusterId_count,
                               int dims) {
    int clusterId = device_clusterId_of_point[blockIdx.x];
    if(threadIdx.x % dims == 0) {
        atomicAdd(clusterId_count+clusterId,1);
    }
    atomicAdd(current_centroids+clusterId*dims + threadIdx.x,device_input_vals[blockIdx.x*dims + threadIdx.x]);
}

__global__ void average_centroids(double* current_centroids, int* clusterId_count, int dims) {
    current_centroids[dims * blockIdx.x + threadIdx.x] /= clusterId_count[blockIdx.x];
}


void average_labeled_centroids(double* device_input_vals,
                                double* current_centroids,
                                int* device_clusterId_of_point,
                               int n_vals,
                               int num_cluster,
                               int dims){
      cudaMemset(current_centroids, 0, num_cluster*dims*sizeof(double));
      int* clusterId_count;
      cudaMalloc(&clusterId_count, num_cluster * sizeof(int));
      cudaMemset(clusterId_count, 0, num_cluster * sizeof(int));
      add_to_centroids<<<n_vals, dims>>>(device_clusterId_of_point,
                                                        device_input_vals,
                                                        current_centroids,
                                                        clusterId_count,
                                                        dims);      
      average_centroids<<<num_cluster, dims>>>(current_centroids,clusterId_count, dims);
}

__global__ void compute_sqaured_distance(double* centroids,double* old_centroids,double* sqaured_distance,int dims){
    int idx = dims * blockIdx.x + threadIdx.x;
    double single_sqaured_distance = (centroids[idx] - old_centroids[idx]) * (centroids[idx] - old_centroids[idx]);
    atomicAdd(sqaured_distance+blockIdx.x, single_sqaured_distance);
}

__global__ void check_converged(int* result, double* sqaured_distance, double threshold){
    atomicAnd(result, sqrtf(sqaured_distance[blockIdx.x]) < threshold);
}

bool converged(double* centroids,
               double* old_centroids,
               int num_cluster,
               int dims,
               double threshold){
    double* sqaured_distance;
    cudaMalloc(&sqaured_distance, num_cluster * sizeof(double));
    cudaMemset(sqaured_distance, 0, num_cluster * sizeof(double));
    compute_sqaured_distance<<<num_cluster, dims>>>(centroids, old_centroids, sqaured_distance, dims);
    int* result;
    cudaMalloc(&result, sizeof(int));
    cudaMemset(result, 1, sizeof(int));
    check_converged<<<num_cluster,1>>>(result, sqaured_distance, threshold);
    
    int host_result;
    cudaMemcpy(&host_result, result, sizeof(int),
               cudaMemcpyDeviceToHost);
    
    return host_result;
}

void find_nearest_centroids(double* device_input_vals,
                            double* current_centroids,
                            int* device_clusterId_of_point,
                            int n_vals,
                            int num_cluster,
                            int dims){
 
    double* distance_of_dims;
    cudaMalloc(&distance_of_dims, num_cluster * n_vals * sizeof(double));
    cudaMemset(distance_of_dims, 0, num_cluster * n_vals * sizeof(double));
    calculate_distance_of_dims<<<n_vals * num_cluster, dims>>>(distance_of_dims,
                                                                   device_input_vals,
                                                                   current_centroids,
                                                                   n_vals,
                                                                   num_cluster,
                                                                   dims);

    get_nearest<<<n_vals,1>>>(device_clusterId_of_point, distance_of_dims, n_vals, num_cluster, dims);
}



int kmeans(const struct options_t& opts,
            int n_vals,
            double* centroid,
            double* input_vals,
            int* clusterId_of_point,
            float* elapsed_time, float* data_transfer_time){
    cudaEvent_t start, stop, job_start, job_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&job_start);
    cudaEventCreate(&job_stop);
    cudaEventRecord(start);    
    int iterations = 0;
    double* old_centroids;
    double* current_centroids;
    cudaMalloc(&old_centroids, opts.dims * opts.num_cluster * sizeof(double));
    cudaMalloc(&current_centroids, opts.dims * opts.num_cluster * sizeof(double));
    cudaMemcpy(current_centroids, centroid,  opts.dims * opts.num_cluster * sizeof(double), cudaMemcpyHostToDevice);
    
    double* device_input_vals;
    cudaMalloc(&device_input_vals, opts.dims * n_vals * sizeof(double));
    cudaMemcpy(device_input_vals,input_vals,  opts.dims * n_vals * sizeof(double), cudaMemcpyHostToDevice);
    
    int* device_clusterId_of_point;
    cudaMalloc(&device_clusterId_of_point, n_vals * sizeof(int));
    cudaMemcpy(device_clusterId_of_point,clusterId_of_point,  n_vals * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(job_start);
    bool done = false;
    while(!done) {
        cudaMemcpy(old_centroids, current_centroids, opts.dims * opts.num_cluster * sizeof(double), cudaMemcpyDeviceToDevice);
        
        find_nearest_centroids(device_input_vals, current_centroids, device_clusterId_of_point, n_vals, opts.num_cluster, opts.dims);
        
        average_labeled_centroids(device_input_vals, current_centroids, device_clusterId_of_point, n_vals, opts.num_cluster, opts.dims);
        
        iterations++;
        done = iterations > opts.max_num_iter || converged(current_centroids, old_centroids, opts.num_cluster, opts.dims, opts.threshold);
    }
    
    cudaEventRecord(job_stop);
    cudaEventSynchronize(job_stop);
    cudaMemcpy(centroid, current_centroids,  opts.dims * opts.num_cluster * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(clusterId_of_point,device_clusterId_of_point,  n_vals * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(old_centroids);
    cudaFree(current_centroids);
    cudaFree(device_input_vals);
    cudaFree(device_clusterId_of_point);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);
    
    float job_time;
    cudaEventElapsedTime(&job_time, job_start, job_stop);
    *data_transfer_time = *elapsed_time - job_time;
    return iterations;
}