#include "kmeans_init.h"
#include "kmeans_seq.h"
#include "argparse.h"
#include "io.h"

#if defined(KMEANS_CUDA_BASIC) || defined(KMEANS_CUDA_SHMEM)
    #include "kmeans_cuda.h"
#endif

using namespace std;
    
int main(int argc, char **argv)
{
    struct options_t opts;
    get_opts(argc, argv, &opts);  

    int n_vals;
    float* input_vals;
    read_file(&n_vals, &input_vals, &opts);
    float* centroids = init_centroids(&opts, n_vals, input_vals);
    
    int iterations        = 0;
    int copy_milliseconds = 0;
    int exec_milliseconds = 0; 

#if defined(KMEANS_CUDA_BASIC) || defined(KMEANS_CUDA_SHMEM)
    printCudaInfo();
#endif 
    
	auto total_start = std::chrono::high_resolution_clock::now(); 

#ifdef KMEANS_SEQ
    kmeans_seq(&centroids, &iterations, &exec_milliseconds, &opts, input_vals, n_vals);
#endif
#ifdef KMEANS_THRUST
    
#endif 
#ifdef KMEANS_CUDA_BASIC
    kmeans_cuda_basic(&centroids, &iterations, &copy_milliseconds, &exec_milliseconds, &opts, input_vals, n_vals);
#endif 
#ifdef KMEANS_CUDA_SHMEM
    kmeans_cuda_shmem(&centroids, &iterations, &copy_milliseconds, &exec_milliseconds, &opts, input_vals, n_vals);
#endif
    
	auto total_end  = std::chrono::high_resolution_clock::now();
    auto total_diff = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    int total_milliseconds = total_diff.count(); 
	
	if (opts.print_total_time) {
		std::cout << "total time "<< total_milliseconds << std::endl;
	}
    if (opts.print_copy_time){
        std::cout << "data copy time "<< copy_milliseconds << std::endl;
    }
    
    std::cout << iterations <<"," << exec_milliseconds/iterations << std::endl;
    if (opts.output_centroids) {
        for (int i = 0; i < opts.k_clusters; i++){
            printf("%d ", i);
            for (int d = 0; d < opts.n_dims; d++) {
                printf("%lf ", centroids[(i * opts.n_dims) + d]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int i = 0; i < n_vals; i++) {
            printf(" %d", find_min_dist_centroid(&opts, i, input_vals, centroids));
        }
    }
}