#include <iostream>
#include <argparse.h>
#include <io.h>
#include <chrono>
#include <cstring>
#include <kmeans_seq.h>

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
        (clusters)[i] = (double*) malloc((opts.n_dims) * sizeof(double));
        // you should use the proper implementation of the following
        // code according to your data structure
        for (int j = 0 ; j < opts.n_dims; j++)
        {
            (clusters)[i][j] = input_vals[index][j];
            //std::cout<<clusters[i][j]<<" "<<input_vals[index][j]<<"id "<<index<<std::endl;
        }
        //std::cout<<index<<",";
    }
    //std::cout<<"sud seg fault"<<endl;
    /*
    srand(opts.seed);
    for (int i = 0; i < opts.n_clusters; i++)
    {
        //std::cout<<(*input_vals)[i]<<endl;
        (clusters)[i] = (double*) malloc((opts.n_dims) * sizeof(double));
        int id = rand() % n_vals;
        for (int j = 0 ; j < opts.n_dims; j++)
        {
            (clusters)[i][j] = input_vals[id][j];
            //std::cout<<clusters[i][j]<<" "<<input_vals[id][j]<<"id "<<id<<endl;
        }
    }
    */
    //void kmeans_seq(int n_vals,int n_dims,int* cluster_id, double** input_vals, int max_iter, int threshold, double*** clusters, int n_clusters )
    kmeans_seq(n_vals, opts.n_dims, cluster_id, input_vals, opts.max_iter, opts.threshold, &clusters, opts.n_clusters);
    
    if(opts.control)
    {
        for (int j = 0 ; j < opts.n_clusters; j++)
        {
            printf("%d ", j);
            for (int d = 0; d < opts.n_dims; d++)
                printf("%lf ", clusters[j][d]);
            printf("\n");
        }
    }
    else
    {
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", cluster_id[p]);
    }
            //std::cout<<cluster_id[j]<<" sud"<<std::endl;
    //std::cout<<*input_vals<<endl;
    
    //fill_args(ps_args, opts.n_threads, n_vals, input_vals, output_vals,
    //    opts.spin, scan_operator, opts.n_loops);

    // Start timer
    
    
    
    
    //End timer and print out elapsed
    
    // Write output data
    //write_file(&opts, &(ps_args[0]));

    // Free other buffers
    
}
