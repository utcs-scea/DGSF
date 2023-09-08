#include <iostream>
#include <argparse.h>
#include <io.h>
#include <chrono>
#include <cstring>
#include <cmath>
#include "kmeans.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Setup args & read input data
    REAL   *input_dataset;
    int    *output_labels;
    REAL   *output_centroids;
    read_file(&opts, &input_dataset, &output_labels, &output_centroids);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // kmeans algorithm
    float time_loops_in_ms;
    int iter_to_converge;

#ifdef KMEANS_SEQ
    kmeans_seq(
        &opts, &input_dataset, 
        &output_labels, &output_centroids,
        &time_loops_in_ms, &iter_to_converge);
#endif
#ifdef KMEANS_THRUST
    kmeans_thrust(
        &opts, &input_dataset, 
        &output_labels, &output_centroids,
        &time_loops_in_ms, &iter_to_converge);
#endif
#ifdef KMEANS_CUDA
    kmeans_cuda(
        &opts, &input_dataset, 
        &output_labels, &output_centroids,
        &time_loops_in_ms, &iter_to_converge);
#endif
#ifdef KMEANS_CUDA_SHMEM
    kmeans_cuda_shmem(
        &opts, &input_dataset, 
        &output_labels, &output_centroids,
        &time_loops_in_ms, &iter_to_converge);
#endif

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float time_end_to_end_in_ms = diff.count()/1000.0;
    if(opts.end2end){
        printf("end to end: %lf\n", time_end_to_end_in_ms);
    }

    // Output
    printf("%d,%lf\n", iter_to_converge, time_loops_in_ms/float(iter_to_converge));
    if (opts.control){
        //if -c is specified, your program should output the centroids of final clusters in the following form
        for (int k = 0; k < opts.num_cluster; k++){
            printf("%d ", k);
            for (int d = 0; d < opts.dims; d++){
                printf("%lf ", output_centroids[k*opts.dims+d]);
            }
            printf("\n");
        }
    }else{
        // If -c is not specified to your program, it needs to write points assignment, i.e. the final cluster id for each point, to STDOUT in the following form
        // Output Points Labels
        printf("clusters:");
        for (int p = 0; p < opts.num_points; p++){
            printf(" %d", output_labels[p]);
        }
    }

    // Free other buffers
    free(input_dataset);
    free(output_labels);
    free(output_centroids);
}
