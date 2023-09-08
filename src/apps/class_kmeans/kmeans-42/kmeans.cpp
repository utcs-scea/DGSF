#include <stdio.h>
#include <assert.h>
#include <getopt.h>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "kmeans.h"
#include <cmath>
#include <cuda_runtime.h>
#include  <algorithm>
#include <cmath>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void printHelp()
{
    printf("kmeans usage:\n");
    printf("    -k (num_cluster: an integer specifying the number of clusters)\n");
    printf("    -d  (dims: an integer specifying the dimension of the points)\n");
    printf("    -i  (inputfilename: a string specifying the input filename)\n");
    printf("    -m (max_num_iter: an integer specifying the maximum number of iterations)\n");
    printf("    -t (threshold: a double specifying the threshold for convergence test.)\n");
    printf("    -c (output the centroids of all clusters if used, output the labels of all points if not \n");
    printf("    -s (seed: an integer specifying the seed for rand())\n");
    printf("    -y (type: an integer specifying the type of algorithm to run (seq, thrust, cuda_basic, cuda_shmem)\n");
    printf("    -n (num_threads: an integer specifying num threads per block to use in cuda implementations)");
}

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        printHelp();
        exit(0);
    }
    
    opts->c = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"c", no_argument, NULL, 'c'},
        {"type", required_argument, NULL, 'y'}
    };

    int ind, a;
    while ((a = getopt_long(argc, argv, "k:d:i:m:t:s:y:c", l_opts, &ind)) != -1)
    {
        switch (a)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->inputfilename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->c = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'y':
            opts->type = atoi((char *)optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

void read_file(struct options_t* args,
               int*              n_vals,
               double**          input_vals) {

  	// Open file
	std::ifstream in;
	in.open(args->inputfilename);
	// Get num vals
	in >> *n_vals;

	// Alloc input array
	*input_vals = (double*) malloc(*n_vals * args->dims * sizeof(double));
    int placeholder;

	// Read input vals
	for (int i = 0; i < *n_vals; ++i) {
        for(int n = 0; n < args->dims + 1; n++){
            if(n != 0){
                in >> (*input_vals)[(i*args->dims)+(n-1)];
            } else {
                in >> placeholder;
            }
        }
	}
}

double euclidean_distance(double* x, double* y, int size){
    double total = 0;
    for(int i=0; i<size; i++){
        total += pow(x[i] - y[i], 2);
    }
    return pow(total, 0.5);
}

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

void print_arr(double* centroids, int dims, int num_cluster){
    for(int i = 0; i < num_cluster; i++){
        printf("%d ", i);
        for (int j = 0; j < dims; j++)
            printf("%lf ", centroids[(i * dims) + j]);
        printf("\n");
    }
}

void sequential_kmeans(int n_vals, int dims, int num_centroids, double* centroids, double *input_vals, double threshold, int max_num_iter, bool c){
    // Start timer
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double old_centroids[dims*num_centroids];
    int clusterId_of_point[n_vals];
    for(int i=0; i<(dims*num_centroids); i++){
        old_centroids[i] = centroids[i];
    }
    int iter_to_converge = 0;
    for (int step=0; step < max_num_iter; step++){
        iter_to_converge++;
        double sums[num_centroids*dims]={0};
        double counts[num_centroids]={0};

        for(int i=0; i<n_vals; i++ ){
            int min_index = 0;
            double min_distance = std::numeric_limits<double>::infinity();
            // nearest centroid
            for (int j=0; j<num_centroids; j++){
                double new_distance = euclidean_distance(&centroids[j*dims], &input_vals[i*dims], dims);
                if (new_distance <= min_distance){
                    min_index = j;
                    min_distance = new_distance;
                }
            }
            for(int j=0; j<dims; j++){
                sums[min_index * dims +j] += input_vals[(i*dims) + j];
            }
            counts[min_index] += 1; 
            clusterId_of_point[i] = min_index;
        }
        
       for(int i=0; i<num_centroids; i++){
            if (counts[i] == 0) continue;
            for(int j=0; j< dims; j++){
                centroids[i*dims + j] = sums[i*dims +j] / counts[i];
            }
        }
        
        // Compare the centroid values with the old ones
        bool close_enough = true;
        for (int i=0; i<dims*num_centroids; i++){
            if (std::abs(centroids[i] - old_centroids[i]) > threshold){
                close_enough = false;
            };
            old_centroids[i] = centroids[i];
        }
        if (close_enough) break;
        
    }
    // End timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    printf("%d,%lf\n", iter_to_converge, fp_ms.count()/iter_to_converge);
    if(c == true){
        print_arr(centroids, dims, num_centroids);
    } else {
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", clusterId_of_point[p]);
    }
}


int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    
    // read input file
    double *input_vals;
    int numpoints;
    read_file(&opts, &numpoints, &input_vals);
    
    // initialize centroids
    kmeans_srand(opts.seed); // cmd_seed is a cmdline arg
    double *centroids = (double*) malloc(opts.num_cluster * numpoints * opts.dims * sizeof(double));
    for (int i=0; i< opts.num_cluster; i++){
        int index = kmeans_rand() % numpoints;
//         std::cout << index << std::endl;
        for(int j=0; j<opts.dims; j++){
            centroids[i*opts.dims + j] = input_vals[(index*opts.dims) + j];
        }        
    }
    
    // sequential CPU implementation
    if(opts.type == 1){        
        sequential_kmeans(numpoints, opts.dims, opts.num_cluster, centroids, input_vals, opts.threshold, opts.max_num_iter, opts.c);  
    }
    // parallel thrust implementation
    else if(opts.type == 2){
        parallel_thrust_kmeans(numpoints, opts.dims, opts.num_cluster, centroids, input_vals, opts.threshold, opts.max_num_iter, opts.c);
    }
    // parallel cuda basic implementation
    else if(opts.type == 3) {
        cuda_basic_kmeans(numpoints, opts.dims, opts.num_cluster, centroids, input_vals, opts.threshold, opts.max_num_iter, opts.c, 32);
    }
    // parallel cuda shared memory implementation
    else if(opts.type == 4) {
        cuda_shmem_kmeans(numpoints, opts.dims, opts.num_cluster, centroids, input_vals, opts.threshold, opts.max_num_iter, opts.c, 32);
    }
    
}