#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <chrono>
#include "argparse.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}


__device__ double getDistanceSq(double* point1, double* point2, int dim) {
    double distance_sq = 0;
    for(int i=0; i < dim; i++) {
        distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
    }
    
    return distance_sq;
}

__device__ int getNearestCentroid(double* cuda_input_vals,
                                  int row_id,
                                  double* cuda_k_centroids,
                                  int num_cluster,
                                  int dims) {
    int nearest_index = 0;
    double smallest_dist =  getDistanceSq(cuda_input_vals + row_id,
                                          cuda_k_centroids + nearest_index*dims,
                                          dims);
    for(int i=1; i < num_cluster; i++) {
        double temp = getDistanceSq(cuda_input_vals + row_id,
                                    cuda_k_centroids + i*dims,
                                    dims);
        if (temp < smallest_dist) {
            nearest_index = i;
            smallest_dist = temp;
        }
    }
    
    return nearest_index;
}


__device__ double d_atomicAdd(double* address, double val)
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

__global__ void setCluster(double* cuda_input_vals,
                        double* cuda_k_centroids,
                        double* cuda_k_centroids_sum,
                        int* cuda_num_points_each_cluster,
                        int* cuda_clusterid_of_points,
                        int num_vals,
                        int num_cluster,
                        int dims) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_vals) {
        return;
    }
    
    int row_id = index * dims;
    int centroid_index = getNearestCentroid(cuda_input_vals,
                                            row_id,
                                            cuda_k_centroids,
                                            num_cluster,
                                            dims);
     cuda_clusterid_of_points[index] = centroid_index;
    
     //printf("chetan index:%d %d %d %d centroid:%d\n", blockIdx.x * blockDim.x + threadIdx.x, blockIdx.x, blockDim.x, threadIdx.x, centroid_index);
    
     atomicAdd(&cuda_num_points_each_cluster[centroid_index], 1);
    
    
     for(int z=0; z < dims; z++) {
       d_atomicAdd(&cuda_k_centroids_sum[centroid_index*dims + z], cuda_input_vals[row_id + z]);
     }
    
}



__global__ void setClusterSharedMem(double* cuda_input_vals,
                        double* cuda_k_centroids,
                        double* cuda_k_centroids_sum,
                        int* cuda_num_points_each_cluster,
                        int* cuda_clusterid_of_points,
                        int num_vals,
                        int num_cluster,
                        int dims) {
    
    extern __shared__ double cuda_shared_k_centroids[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_vals) {
        return;
    }
    
    if (threadIdx.x < num_cluster) {
        for(int i=0; i < dims; i++) {
            cuda_shared_k_centroids[threadIdx.x * dims + i] = cuda_k_centroids[threadIdx.x * dims + i];
        }
    }
    
    __syncthreads();
    
    int row_id = index * dims;
    int centroid_index = getNearestCentroid(cuda_input_vals,
                                            row_id,
                                            cuda_shared_k_centroids,
                                            num_cluster,
                                            dims);
     cuda_clusterid_of_points[index] = centroid_index;
    
     //printf("chetan index:%d %d %d %d centroid:%d\n", blockIdx.x * blockDim.x + threadIdx.x, blockIdx.x, blockDim.x, threadIdx.x, centroid_index);
    
     atomicAdd(&cuda_num_points_each_cluster[centroid_index], 1);
    
    
     for(int z=0; z < dims; z++) {
       d_atomicAdd(&cuda_k_centroids_sum[centroid_index*dims + z], cuda_input_vals[row_id + z]);
     }
    
}




__global__ void computeCentroid(double* cuda_k_centroids,
                               double* cuda_k_centroids_sum,
                               int* cuda_num_points_each_cluster,
                               int num_cluster,
                               int dims,
                               int* cuda_has_converged,
                               double threshold) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_cluster) {
        //printf("unexpected index accessed\n");
        return;
    }
    
    int total_points = cuda_num_points_each_cluster[index];

    if (total_points == 0) {
        //printf("no points in the cluster\n");
        return;
    }
    
    double l2_norm = 0;
    for(int i=0; i < dims; i++) {
        double temp = cuda_k_centroids_sum[index*dims + i]/total_points;
        l2_norm += (temp - cuda_k_centroids[index*dims + i]) * (temp - cuda_k_centroids[index*dims + i]);
        cuda_k_centroids[index*dims + i] = temp;
    }
    
    if (l2_norm > threshold*threshold) {
        cuda_has_converged[index] = 0;
    } else {
        cuda_has_converged[index] = 1;
    }    
}


void read_file(struct options_t& args,
               int*              n_vals,
               double***         input_vals,
               double*** k_centroids,
               double*** k_old_centroids) {

    // Open file
    std::ifstream in;
    in.open(args.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array
    *input_vals = (double**)malloc((*n_vals) * sizeof(double*));
    (*input_vals)[0] = (double*)malloc((*n_vals)*(args.dims)  * sizeof(double));
    for(int i =1; i < *n_vals; i++) {
        (*input_vals)[i] = (*input_vals)[i-1] + args.dims;
    }

    // Read input vals
    for (int i = 0; i < *n_vals; ++i) {
        int temp_index; // unused for now
        in >> temp_index;
        for(int j=0; j < args.dims; j++) {
            in >> (*input_vals)[i][j];
		}
    }


	// k centroids array init
    *k_centroids = (double**)malloc((args.num_cluster) * sizeof(double*));
    (*k_centroids)[0] = (double*)malloc((args.num_cluster)*(args.dims)  * sizeof(double));
    for(int i =1; i < args.num_cluster; i++) {
        (*k_centroids)[i] = (*k_centroids)[i-1] + args.dims;
    }
    

    // k old centroids array init
    *k_old_centroids = (double**)malloc((args.num_cluster) * sizeof(double*));
    (*k_old_centroids)[0] = (double*)malloc((args.num_cluster)*(args.dims)  * sizeof(double));
    for(int i =1; i < args.num_cluster; i++) {
        (*k_old_centroids)[i] = (*k_old_centroids)[i-1] + args.dims;
    }
    
    
    kmeans_srand(args.seed);	
    for (int i=0; i< args.num_cluster; i++) {
        int index = kmeans_rand() % (*n_vals);
        for(int j=0; j < args.dims; j++) {
            (*k_centroids)[i][j] = (*input_vals)[index][j];
        }
    }
}



int main(int argc, char** argv) {
    
    struct options_t options;
    get_opts(argc,
             argv,
             &options);
    int num_vals = 0;
    double** data_points;
    double** k_centroids;
    double** k_old_centroids;
    read_file(options,
              &num_vals,
              &data_points,
              &k_centroids,
              &k_old_centroids);

    
    double* cuda_data_points;
    cudaMalloc((void**)&cuda_data_points, num_vals * options.dims * sizeof(double));
    cudaMemcpy(cuda_data_points, data_points[0], num_vals * options.dims * sizeof(double), cudaMemcpyHostToDevice);
    
    
    double* cuda_k_centroids;
    cudaMalloc((void**)&cuda_k_centroids, options.num_cluster * options.dims * sizeof(double));
    cudaMemcpy(cuda_k_centroids, k_centroids[0], options.num_cluster * options.dims * sizeof(double), cudaMemcpyHostToDevice);
    

    double* cuda_k_centroids_sum;
    cudaMalloc((void**)&cuda_k_centroids_sum, options.num_cluster * options.dims * sizeof(double));    
    
    int *cuda_num_points_each_cluster;
    cudaMalloc((void**)&cuda_num_points_each_cluster, options.num_cluster * sizeof(int));

 
    int *cuda_clusterid_of_points;
    cudaMalloc((void**)&cuda_clusterid_of_points, num_vals * sizeof(int));
    
    int* has_converged = (int*)malloc(sizeof(int) * options.num_cluster);
    int* cuda_has_converged;
    cudaMalloc((void**)&cuda_has_converged, num_vals * sizeof(int));
    
    int num_threads = (options.num_cluster > 512) ? options.num_cluster : 512;
    int num_blocks = (num_vals + num_threads - 1)/num_threads;
    //printf("num_threads:%d num_blocks:%d\n", num_threads, num_blocks);

    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start);
    int iter=0;
    for(; iter < options.max_num_iter; iter++) {
        cudaMemset(cuda_k_centroids_sum, 0, options.num_cluster * options.dims * sizeof(double));
        cudaMemset(cuda_num_points_each_cluster, 0, sizeof(int) * options.num_cluster);
        
        if (options.use_shared_mem == 1) {
            setClusterSharedMem<<<num_blocks, num_threads, options.num_cluster * options.dims * sizeof(double)>>>(cuda_data_points,
                           cuda_k_centroids,
                           cuda_k_centroids_sum,
                           cuda_num_points_each_cluster,
                           cuda_clusterid_of_points,
                           num_vals,
                           options.num_cluster,
                           options.dims);
        } else {
            setCluster<<<num_blocks, num_threads>>>(cuda_data_points,
                           cuda_k_centroids,
                           cuda_k_centroids_sum,
                           cuda_num_points_each_cluster,
                           cuda_clusterid_of_points,
                           num_vals,
                           options.num_cluster,
                           options.dims);
            
        }
        
        //cudaDeviceSynchronize();
        
        computeCentroid<<<1,options.num_cluster>>>(cuda_k_centroids,
                                          cuda_k_centroids_sum,
                                          cuda_num_points_each_cluster,
                                          options.num_cluster,
                                          options.dims,
                                          cuda_has_converged,
                                          options.threshold);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(has_converged, cuda_has_converged, options.num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
        bool converged = true;
        for(int z=0; z < options.num_cluster; z++) {
            if (has_converged[z] == 0) {
                converged = false;
            }
        }
        
        if (converged) {
            break;
        }
        

        
    }
    

    cudaEventRecord(cu_stop);     
    cudaEventSynchronize(cu_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);    
    printf("%d,%lf\n", (iter+1), milliseconds/(iter+1));
    
    cudaMemcpy(k_centroids[0], cuda_k_centroids, options.num_cluster * options.dims * sizeof(double), cudaMemcpyDeviceToHost);
    int *clusterid_of_points = (int *)malloc(num_vals * sizeof(int));
    cudaMemcpy(clusterid_of_points, cuda_clusterid_of_points, num_vals * sizeof(int), cudaMemcpyDeviceToHost);

    
    
    if (options.print_centroid) {
        for (int i = 0; i < options.num_cluster; i++) {
            printf("%d ", i);
            for (int j=0; j < options.dims; j++) {
                printf("%lf ", k_centroids[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < num_vals; p++) {
            printf(" %d", clusterid_of_points[p]);
        }
    }
    
    
    
    free(data_points[0]);
    free(data_points);
    free(k_centroids[0]);
    free(k_centroids);
    free(k_old_centroids[0]);
    free(k_old_centroids);
    free(clusterid_of_points);
    //free(num_points_each_cluster);
    
	
    return 0;
}
