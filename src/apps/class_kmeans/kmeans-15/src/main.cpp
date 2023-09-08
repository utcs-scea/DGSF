#include <iostream>
#include <argparse.h>
#include <io.h>
#include <chrono>
#include <cstring>
#include <kmeans_seq.h>
#include <stdio.h>
//#include <kmeans_thrust.cu>

//void kmeans_thrust(int n_vals,int n_dims,int* cluster_id, double* input_vals, int max_iter, double threshold, double* clusters, int n_clusters );
void kmeans_kernel_launch(int n_vals,int n_dims,int* cluster_id, float* input_vals, int max_iter, double threshold, float* clusters, int n_clusters );
    
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
    //std::cout<<seed<<" seed"<<std::endl;
}
    

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    
    // Setup args & read input data
    //prefix_sum_args_t *ps_args = alloc_args(opts.n_threads);
    int n_vals;
    double **input_vals;
    int* cluster_id;
    double **clusters;
    read_file(&opts, &n_vals, &input_vals, &opts.n_dims);
    clusters = (double**) malloc((opts.n_clusters) * sizeof(double));
    cluster_id = (int*) malloc((n_vals) * sizeof(int));
    //std::cout<<"sud seed "<<opts.seed<<" max "<<opts.max_iter<<" dim "<<opts.n_dims<<" n_cl "<<opts.n_clusters<<" thr "<<opts.threshold<<std::endl;
    //kmeans_seq(n_vals, n_dims, &cluster_id, &input_vals);
    kmeans_srand(opts.seed); // cmd_seed is a cmdline arg
    for (int i=0; i<opts.n_clusters; i++)
    {
        int index = kmeans_rand() % n_vals;
        //cout<<"rand index "<<index<<endl;
        (clusters)[i] = (double*) malloc((opts.n_dims) * sizeof(double));
        for (int j = 0 ; j < opts.n_dims; j++)
        {
            (clusters)[i][j] = input_vals[index][j];
        }
    }
    float* input_vals_1d;
    float* clusters_1d;
    input_vals_1d = (float*) malloc((n_vals)* opts.n_dims * sizeof(float));
    clusters_1d = (float*) malloc((opts.n_clusters)* opts.n_dims * sizeof(float));
    
    for (int i = 0; i < n_vals; i++)
    {
        for (int j = 0 ; j < opts.n_dims; j++)
        {
            input_vals_1d[i*opts.n_dims+j] = input_vals[i][j];
        }
    }
    
    for (int i = 0; i < opts.n_clusters; i++)
    {
        for (int j = 0 ; j < opts.n_dims; j++)
        {
            clusters_1d[i*opts.n_dims+j] = clusters[i][j];
        }
    }
    
    //kmeans_seq(n_vals, opts.n_dims, cluster_id, input_vals, opts.max_iter, opts.threshold, &clusters, opts.n_clusters);
    
    //Thrust implementation
    //kmeans_thrust(n_vals, opts.n_dims, cluster_id, input_vals_1d, opts.max_iter, opts.threshold, clusters_1d, opts.n_clusters);
    
    //cout<<"kernel launch"<<endl;
   
    
    kmeans_kernel_launch(n_vals, opts.n_dims, cluster_id, input_vals_1d, opts.max_iter, opts.threshold, clusters_1d, opts.n_clusters);
    /*
    if(opts.control)
    {
        for (int j = 0 ; j < opts.n_clusters; j++)
        {
            printf("%d ", j);
            for (int d = 0; d < opts.n_dims; d++)
                printf(" %f ", clusters[j][d]);
            printf("\n");
        }
    }
    else
    {
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", cluster_id[p]);
        printf("\n");
    }
    */
   
    if(opts.control)
    {
        for (int j = 0 ; j < opts.n_clusters; j++)
        {
            printf("%d ", j);
            for (int d = 0; d < opts.n_dims; d++)
                printf(" %f ", clusters_1d[j*opts.n_dims +d]);
            printf("\n");
        }
    }
    else
    {
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", cluster_id[p]);
        printf("\n");
    }
    
    
    
    
    
    //End timer and print out elapsed
    
    // Write output data
    //write_file(&opts, &(ps_args[0]));

    // Free other buffers
    
}
