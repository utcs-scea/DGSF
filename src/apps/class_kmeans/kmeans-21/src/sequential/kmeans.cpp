#include <iostream>
#include <chrono>
#include "../common/common_functions.h"

int main(int argc, char **argv) {
    struct options_t opts;
    int n_vals;
    float *points, *centroids, *old_centroids;
    get_opts(argc, argv, &opts);
    read_file(&opts, &n_vals, &points);
    get_initial_centroids(&opts, &n_vals, points, &centroids);

    old_centroids = (float *) malloc(opts.num_cluster * opts.dims * sizeof(float));
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    int *cluster_mapping = (int *) malloc(n_vals * sizeof(int));
    int iters = 0;
    for (; iters < opts.max_num_iter; ++iters) {
        assign_clusters(&opts, points, centroids, &n_vals, cluster_mapping);
        swap_centroids(&old_centroids, &centroids);
        recompute_centroids(&opts, cluster_mapping, points, centroids, &n_vals);
        if (test_convergence(&opts, old_centroids, centroids)){
            ++iters;
            break;
        }
    }
    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("%d,%lf\n", iters, (double) diff.count() / (double) iters / 1000.0);
    if (opts.output_centroids) {
        print_cluster_centroids(&opts, centroids);
    } else {
        print_cluster_mappings(&n_vals, cluster_mapping);
    }
    
    free(centroids);
    free(old_centroids);
    free(points);
    free(cluster_mapping);
    return 0;
}