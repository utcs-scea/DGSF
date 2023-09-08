#include <kmeans.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <an integer specifying the number of clusters.>" << std::endl;
        std::cout << "\t--dims or -d <an integer specifying the dimension of the points.>" << std::endl;
        std::cout << "\t--inputfilename or -i <a string specifying the input filename.>" << std::endl;
        std::cout << "\t--max_num_iter or -m <an integer specifying the maximum number of iterations.>" << std::endl;
        std::cout << "\t--threshold or -t <a double specifying the threshold for convergence test.>" << std::endl;
        std::cout << "\t--seed or -s <an integer specifying the seed for rand().>" << std::endl;
        std::cout << "\t[Optional] --centroid or -c :a flag to control the output of your program." << std::endl;
        exit(0);
    }

    opts->centroid = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"centroid", no_argument, NULL, 'c'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:c", l_opts, &ind)) != -1)
    {
        switch (c)
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
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'c':
            opts->centroid = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

void read_file(const struct options_t& args,
               int*                    n_vals,
               double**                input_vals) {
    // Open file
	std::ifstream in;
	in.open(args.inputfilename);
	// Get num vals
	in >> *n_vals;
    
    *input_vals = (double *)malloc(args.dims * (*n_vals) * sizeof(double));
    
    std::string fst_line;
    std::getline(in, fst_line);
    for(int i = 0; i < *n_vals; ++i){
        std::string line;
        std::getline(in, line);
        std::istringstream iss(line);
        
        double idx;
        iss >> idx;
        
        for(int j = 0; j < args.dims; ++j){
            double number;
            iss >> number;
            (*input_vals)[i*args.dims + j] = number;
        }
    }
}

void generate_initial_centroids(const struct options_t& args,
                                int n_vals,
                                double* centroid,
                                double* input_vals){
    unsigned long int next = 1;
    unsigned long kmeans_rmax = 32767;
    
    next = args.seed;
    
    for (int i=0; i<args.num_cluster; i++){
        next = next * 1103515245 + 12345;
        int kmeans_rand = (unsigned int)(next/65536) % (kmeans_rmax+1);
        
        int index = kmeans_rand % n_vals;
        // you should use the proper implementation of the following
        // code according to your data structure
        for(int j = 0; j < args.dims; ++j){
            centroid[i*args.dims + j] = input_vals[index*args.dims + j];
        }
    }
}


int main(int argc, char **argv) {
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);
    
    int n_vals;
    double* input_vals;
    read_file(opts, &n_vals, &input_vals);
    double* centroid = (double *)malloc(opts.dims * opts.num_cluster * sizeof(double));
    int* clusterId_of_point = (int *)malloc(n_vals * sizeof(int));
    
    float elapsed_time;
    float data_transfer_time;
    generate_initial_centroids(opts, n_vals, centroid, input_vals);
    int iter_to_converge = kmeans(opts, n_vals, centroid, input_vals, clusterId_of_point, &elapsed_time, &data_transfer_time);
    
    
    printf("%d,%lf\n", iter_to_converge, elapsed_time / iter_to_converge);
//     printf("%d,%lf,%lf,%lf\n", iter_to_converge, elapsed_time / iter_to_converge, data_transfer_time / iter_to_converge, data_transfer_time / elapsed_time);
        
    if(opts.centroid){
        for (int clusterId = 0; clusterId < opts.num_cluster; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < opts.dims; d++)printf("%lf ", centroid[clusterId*opts.dims + d]);
            printf("\n");
        }        
    }else{
        printf("clusters:");
        for (int p=0; p < n_vals; p++) printf(" %d", clusterId_of_point[p]);
    }
}