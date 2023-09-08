#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include "loader.h"
#include "argparse.h"
#include "rng.h"
#include "cuda.h"
#include "kmeans_kernel.cuh"

int main(int argc, char** argv) {
    cudaEvent_t start, end, start_exe_only, end_exe_only;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_exe_only);
    cudaEventCreate(&end_exe_only);
    cudaEventRecord(start);
    
    options_t opts;
    get_opts(argc, argv, &opts);
    loader file_loader(opts.dims);
    
    int dims = opts.dims;
    int ncentroids = opts.num_cluster;
    double* points = nullptr;
    double* centroids = nullptr;
    int npoints;
    file_loader.load_as_pointer(opts.inputfilename, &points, npoints);
    int points_size = npoints * dims;
    int centroids_size = ncentroids * dims;
    int cross_size = npoints * ncentroids;
    
    centroids = (double*)malloc(centroids_size * sizeof(double));
    rng randomizer(opts.seed);
    for (int i = 0; i < opts.num_cluster; i++) {
        int idx = randomizer.kmeans_rand() % npoints;
        for (int j = 0; j < dims; j++) {
            centroids[i * dims + j] = points[idx * dims + j];
        }
    }
    
    double *d_points, *d_centroids, *d_distances, *d_new_centroids, *d_convergence_distances;
    int *d_minimum_centroid_ids, *d_centroid_counts, *d_cnv;
    cudaMalloc((void**)&d_points, points_size * sizeof(double));
    cudaMalloc((void**)&d_centroids, centroids_size * sizeof(double));
    cudaMalloc((void**)&d_distances, cross_size * sizeof(double));
    cudaMalloc((void**)&d_new_centroids, centroids_size * sizeof(double));
    cudaMalloc((void**)&d_minimum_centroid_ids, npoints * sizeof(int));
    cudaMalloc((void**)&d_centroid_counts, ncentroids * sizeof(int));
    cudaMalloc((void**)&d_convergence_distances, ncentroids * sizeof(double));
    cudaMalloc((void**)&d_cnv, sizeof(int));
    
    cudaMemcpy(d_points, points, points_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, centroids_size * sizeof(double), cudaMemcpyHostToDevice);
    
    bool is_conv = false;
    int iter = 0;
    int cnv[1];
    
    cudaEventRecord(start_exe_only);
    
    
    while (!is_conv && iter < opts.max_num_iter) {
        cudaMemset(d_centroid_counts, 0, ncentroids * sizeof(int));
        cudaMemset(d_cnv, 0, sizeof(int));
        reset_zero<<<ncentroids, dims>>>(d_new_centroids, centroids_size);
        
        if (opts.cuda_shared) {
            nearest_centroid_shared<<<npoints, ncentroids>>>(d_points, d_centroids, dims, npoints, ncentroids, points_size, centroids_size, cross_size, d_minimum_centroid_ids);
        } else {
            dot<<<npoints, ncentroids>>>(d_points, d_centroids, dims, npoints, ncentroids, points_size, centroids_size, cross_size, d_distances);
            nearest_centroid<<<npoints, 1>>>(d_distances, npoints, ncentroids, cross_size, d_minimum_centroid_ids);            
        }

        count_centroid_id<<<npoints, 1>>>(d_minimum_centroid_ids, npoints, ncentroids, d_centroid_counts);        
        sum_new_centroid_values<<<npoints, dims>>>(d_points, d_minimum_centroid_ids, dims, npoints, points_size, centroids_size, d_new_centroids);
        avg_new_centroid_values<<<ncentroids, dims>>>(d_new_centroids, d_centroid_counts, dims, ncentroids, centroids_size); 
        new_centroid_movement_squared<<<ncentroids, dims>>>(d_centroids, d_new_centroids, dims, ncentroids, centroids_size, d_convergence_distances);
        sqrt_kernel<<<1, ncentroids>>>(d_convergence_distances, ncentroids);
        is_convergent<<<1, ncentroids>>>(d_convergence_distances, opts.threshold, ncentroids, d_cnv);
        cudaMemcpy(cnv, d_cnv, sizeof(int), cudaMemcpyDeviceToHost);
        is_conv = (cnv[0] == ncentroids);
        iter++;
        cudaMemcpy(d_centroids, d_new_centroids, centroids_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    cudaEventRecord(end_exe_only);
    cudaEventSynchronize(end_exe_only);
    cudaEventRecord(end);
    
    float diff = 0;
    cudaEventElapsedTime(&diff, start_exe_only, end_exe_only);
    
    printf("%d,%f\n", iter, diff);
    if (opts.output_centroids) {
        cudaMemcpy(centroids, d_centroids, centroids_size * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < ncentroids; i++) {
            printf("%d ", i);
            for (int j = 0; j < dims; j++) {
                printf("%lf ", centroids[j + dims * i]);
            }
            printf("\n");
        }
    } else {
        int clusters[npoints];
        cudaMemcpy(clusters, d_minimum_centroid_ids, npoints * sizeof(int), cudaMemcpyDeviceToHost);
        printf("clusters:");
        for (int i = 0; i < npoints; i++) {
            printf(" %d", clusters[i]);
        }
    }
    
    if (opts.print_e2e) {
        cudaEventElapsedTime(&diff, start, end);
        printf("%f\n", diff);
    }

    free(points);
    free(centroids);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_new_centroids);
    cudaFree(d_minimum_centroid_ids);
    cudaFree(d_centroid_counts);
    cudaFree(d_convergence_distances);

    return 0;
}