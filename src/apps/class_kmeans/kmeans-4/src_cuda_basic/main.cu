#include <stdio.h>
#include <iostream>
#include <argparse.h>
#include <chrono>
#include <cstring>
#include <math.h>
#include <fstream>

#include <cuda_runtime.h>


// Provided for repeatable RNG
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}


//using namespace std;


void read_file(struct options_t* args,
               int*              n_vals,
               float**           input_vals
) {
    int row_num ;

    // Open file
    std::ifstream in;
    in.open(args->inputfilename);
    // Get num vals
    in >> *n_vals;

    *input_vals = (float*) malloc(*n_vals * args->dims * sizeof(float));

    // Read input vals
    for (int i = 0; i < *n_vals; ++i) {
        in >> row_num;
        for (int j = 0; j < args->dims; ++j) {
            in >> (*input_vals)[i * args->dims + j];
        }
    }
}


__global__
void calc_thresh(float *thresholds, float *centers, float *new_centers, int num_cluster, int dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < num_cluster) {
        for (int j=0; j < dims; ++j) {
            float diff = centers[i*dims + j] - new_centers[i*dims + j];
            val += diff * diff;
        }
        thresholds[i] = val;
    }
    // Consider atomicAdd to a scalar
}


__global__
void nearest_center(
    int *labels,
    float *centers,
    float *input_vals,
    int num_cluster,
    int dims,
    int n_vals
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vals) {
        int best_label = 0;
        float dist = INFINITY;
        float dist_new;
        for (int c=0; c<num_cluster; ++c){
            dist_new = 0;
            for (int j=0; j<dims; ++j){
                float diff = input_vals[i*dims+j] - centers[c*dims + j];
                dist_new += diff * diff;
            }
            if (dist_new < dist) {
                best_label = c;
                dist = dist_new;
            }
        }
        labels[i] = best_label;
    }
}


// Can do one thread per value in new_centers.
// This allows a direct comparison to the implementation using thrust
__global__
void calc_center_sum(
    float *new_centers,
    int *new_centers_n,
    int *labels,
    float *input_vals,
    int num_cluster,
    int dims,
    int n_vals,
    int chunksize
) {
    int c = threadIdx.x / dims;
    int j = threadIdx.x % dims;
    float val = 0.0f;
    int count = 0;
    for (int idx=0; idx < chunksize; ++idx) {
        int i = chunksize * blockIdx.x + idx;
        if (i < n_vals && labels[i] == c) {
            val += input_vals[i*dims + j];
            ++count;
        }
    }
    if (count > 0) {
        atomicAdd(&new_centers[c*dims + j], val);
        if (j == 0) {
            atomicAdd(&new_centers_n[c], count);
        }
    }
}


__global__
void calc_new_center(
    float *new_centers,
    int *new_centers_n,
    int num_cluster,
    int dims
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / dims;
    if (i < num_cluster) {
        new_centers[tid] = new_centers[tid] / (float)new_centers_n[i];
    }
}


int main(int argc, char **argv)
{
            /// auto everything_start = std::chrono::high_resolution_clock::now();

            /// // for timing
            /// float cuda_malloc = 0.0f;
            /// float data_transfer = 0.0f;
            /// float milliseconds = 0.0f;

    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // host vectors
    float *h_input_vals;  // n_vals x opts.dims
    int n_vals;
    read_file(&opts, &n_vals, &h_input_vals);
    float *h_centers = (float*) malloc(opts.num_cluster * opts.dims * sizeof(float));
    float *h_thresholds = (float*) malloc(opts.num_cluster * sizeof(float));
    int *h_labels = (int*) malloc(n_vals * sizeof(int));

    kmeans_srand(opts.seed); // cmd_seed is a cmdline arg
    for (int i=0; i<opts.num_cluster; i++){
        int index = kmeans_rand() % n_vals;
        for (int j = 0; j < opts.dims; ++j) {
            h_centers[i*opts.dims+j] = h_input_vals[index*opts.dims+j];
        }
    }

    // device vectors
    float *d_input_vals;  // n_vals x opts.dims
    float *d_centers;  // opts.num_cluster x opts.dims
    float *d_new_centers;  // opts.num_cluster x opts.dims
    float *d_thresholds;  // opts.num_cluster
    int *d_new_centers_n;  // opts.num_cluster
    int *d_labels;  // n_vals

            /// auto malloc_start = std::chrono::high_resolution_clock::now();

    cudaMalloc((float**)&d_input_vals, n_vals * opts.dims * sizeof(float));
    cudaMalloc((float**)&d_centers, opts.num_cluster * opts.dims * sizeof(float));
    cudaMalloc((float**)&d_new_centers, opts.num_cluster * opts.dims * sizeof(float));
    cudaMalloc((float**)&d_thresholds, opts.num_cluster * sizeof(float));
    cudaMalloc((int**)&d_new_centers_n, opts.num_cluster * sizeof(int));
    cudaMalloc((int**)&d_labels, n_vals * sizeof(int));

            /// auto malloc_stop = std::chrono::high_resolution_clock::now();
            /// auto diff = std::chrono::duration_cast<std::chrono::microseconds>(malloc_stop - malloc_start);
            /// cuda_malloc += diff.count() / 1000.0;

    int iterations = 0;
    float cur_thresh = INFINITY;

    int threads_per_block1 = 512;
    int grid1 = (n_vals + threads_per_block1 - 1) / threads_per_block1;

    int threads_per_block2 = opts.num_cluster * opts.dims;
    int chunksize = 241;  // prime number to limit bank conflicts: 241, 523, etc
    int grid2 = (n_vals + chunksize - 1) / chunksize;

    int threads_per_block3 = 512;
    int grid3 = (opts.num_cluster * opts.dims + threads_per_block3 - 1) / threads_per_block3;

    int threads_per_block4 = 256;
    int grid4 = (opts.num_cluster + threads_per_block4 - 1) / threads_per_block4;

    // Copy to device
            /// cudaEvent_t transfer_timer_start, transfer_timer_stop;
            /// cudaEventCreate(&transfer_timer_start);
            /// cudaEventCreate(&transfer_timer_stop);
            /// cudaEventRecord(transfer_timer_start);

    cudaMemcpy(d_input_vals, h_input_vals, n_vals * opts.dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_centers, h_centers, opts.num_cluster * opts.dims * sizeof(float), cudaMemcpyHostToDevice);

            /// cudaEventRecord(transfer_timer_stop);
            /// cudaEventSynchronize(transfer_timer_stop);
            /// cudaEventElapsedTime(&milliseconds, transfer_timer_start, transfer_timer_stop);
            /// printf("%f\n", milliseconds);
            /// data_transfer += milliseconds;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    while (true) {
        float *temp_centers = d_centers;
        d_centers = d_new_centers;
        d_new_centers = temp_centers;

        nearest_center<<< grid1, threads_per_block1 >>>(
            d_labels,
            d_centers,
            d_input_vals,
            opts.num_cluster,
            opts.dims,
            n_vals
        );

        if (iterations++ >= opts.max_num_iter || cur_thresh <= opts.threshold) {
            break;
        }

        cudaMemset(d_new_centers, 0, opts.num_cluster * opts.dims * sizeof(float));
        cudaMemset(d_new_centers_n, 0, opts.num_cluster * sizeof(int));
        calc_center_sum<<< grid2, threads_per_block2 >>>(
            d_new_centers,
            d_new_centers_n,
            d_labels,
            d_input_vals,
            opts.num_cluster,
            opts.dims,
            n_vals,
            chunksize
        );

        calc_new_center<<< grid3, threads_per_block3 >>>(
            d_new_centers,
            d_new_centers_n,
            opts.num_cluster,
            opts.dims
        );

        calc_thresh<<< grid4, threads_per_block4 >>>(
            d_thresholds,
            d_centers,
            d_new_centers,
            opts.num_cluster,
            opts.dims
        );

        // Very little time is spent here, so no problem coping a small array to the host
                /// cudaEventRecord(transfer_timer_start);

        cudaMemcpy(h_thresholds, d_thresholds, opts.num_cluster * sizeof(float), cudaMemcpyDeviceToHost);

                /// cudaEventRecord(transfer_timer_stop);
                /// cudaEventSynchronize(transfer_timer_stop);
                /// cudaEventElapsedTime(&milliseconds, transfer_timer_start, transfer_timer_stop);
                /// printf("%f\n", milliseconds);
                /// data_transfer += milliseconds;

        cur_thresh = 0.0f;
        for (int i=0; i<opts.num_cluster; ++i) {
            cur_thresh += h_thresholds[i];
        };
        cur_thresh = sqrt(cur_thresh);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float loop_time = 0.0f;
    cudaEventElapsedTime(&loop_time, start, stop);

    // Copy results back to host
            /// cudaEventRecord(transfer_timer_start);

    cudaMemcpy(h_centers, d_centers, opts.num_cluster * opts.dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels, d_labels, n_vals * sizeof(int), cudaMemcpyDeviceToHost);

            /// cudaEventRecord(transfer_timer_stop);
            /// cudaEventSynchronize(transfer_timer_stop);
            /// cudaEventElapsedTime(&milliseconds, transfer_timer_start, transfer_timer_stop);
            /// printf("%f\n", milliseconds);
            /// data_transfer += milliseconds;

            /// auto everything_end = std::chrono::high_resolution_clock::now();
            /// auto end_to_end = std::chrono::duration_cast<std::chrono::microseconds>(everything_end - everything_start);
            /// float total_time = end_to_end.count() / 1000.0;
            /// printf("Total time: %lf\n", total_time);
            /// printf("Data transfer: %lf (%lf%%)\n", data_transfer, 100 * data_transfer / total_time);
            /// printf("Loop body: %lf (%lf%%)\n", loop_time, 100 * loop_time / total_time);
            /// printf("CUDA malloc: %lf (%lf%%)\n", cuda_malloc, 100 * cuda_malloc / total_time);

    // Output (following instructions)
    printf("%d,%lf\n", iterations, loop_time / (float)iterations);
    if (opts.display_centroids) {
        for (int c=0; c < opts.num_cluster; ++c) {
            printf("%d ", c);
            for (int d=0; d < opts.dims; ++d) {
                printf("%lf ", h_centers[c*opts.dims + d]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < n_vals; p++) {
            printf(" %d", h_labels[p]);
        }
    }
    return 0;
}
