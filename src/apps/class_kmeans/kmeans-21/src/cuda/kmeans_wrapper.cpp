#include "../common/common_functions.h"
#include<cuda_runtime.h>
#include "kmeans.h"

int main(int argc, char **argv) {

    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        exit(1);
    }
    
    CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    struct options_t opts;
    int n_vals;
    float *h_points, *h_centroids;
    get_opts(argc, argv, &opts);
    read_file(&opts, &n_vals, &h_points);
    get_initial_centroids(&opts, &n_vals, h_points, &h_centroids);
    
    int point_bytes = n_vals*opts.dims*sizeof(float);
    int centroid_bytes = opts.num_cluster*opts.dims*sizeof(float);
    
    float *d_points, *d_centroids, *d_point_centroid_distances;
    int *d_centroid_assignments, *d_centroid_counts;
    
    CHECK(cudaMalloc((void **)&d_points, point_bytes));
    CHECK(cudaMalloc((void **)&d_centroids, centroid_bytes));
    CHECK(cudaMalloc((void **)&d_point_centroid_distances, n_vals*opts.num_cluster*sizeof(float)));
    CHECK(cudaMalloc((void **)&d_centroid_assignments, n_vals*sizeof(int)));
    CHECK(cudaMalloc((void **)&d_centroid_counts, opts.num_cluster*sizeof(int)));
    
    // transfer data from host to device
    CHECK(cudaMemcpy(d_points, h_points, point_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_centroids, h_centroids, centroid_bytes, cudaMemcpyHostToDevice));
    float milliseconds = 0.0;
    int iters = 0;
    float *h_old_centroids = (float *) malloc(opts.num_cluster * opts.dims * sizeof(float));
    for (; iters < opts.max_num_iter; ++iters) {
        kmeans_iteration_t iter_res = kmeans_cuda(deviceProp, opts, n_vals, d_points, d_centroids, d_centroid_assignments, d_centroid_counts, d_point_centroid_distances, &h_centroids, &h_old_centroids);
        milliseconds += iter_res.time_taken;
        if(iter_res.converged){
            iters++;
            break;
        }
    }
    free(h_old_centroids);
    
    printf("%d,%lf\n", iters, milliseconds / (double) iters);
    if (opts.output_centroids) {
        CHECK(cudaMemcpy(h_centroids, d_centroids, centroid_bytes, cudaMemcpyDeviceToHost));
        print_cluster_centroids(&opts, h_centroids);
    } else {
        int* h_centroid_assignments = (int *) malloc(n_vals * sizeof(int));
        CHECK(cudaMemcpy(h_centroid_assignments, d_centroid_assignments, n_vals * sizeof(int), cudaMemcpyDeviceToHost));
        print_cluster_mappings(&n_vals, h_centroid_assignments);
    }
    
    free(h_points);
    free(h_centroids);
    CHECK(cudaFree(d_points));
    CHECK(cudaFree(d_centroids));  
    CHECK(cudaFree(d_point_centroid_distances));
    CHECK(cudaFree(d_centroid_assignments));
    CHECK(cudaFree(d_centroid_counts));
    
    CHECK(cudaDeviceReset());
    
    return 0;
}