#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k <k_clusters>: no. centroids to output" << std::endl;
        std::cout << "\t-d <n_dims>: no. dimensions per point" << std::endl;
        std::cout << "\t-i <input_file>: path to input points file" << std::endl;
        std::cout << "\t-m <max_num_iter>: maximum no. iterations before declaring convergence" << std::endl;
        std::cout << "\t-c: output the centroid point labels" << std::endl;
        std::cout << "\t-s <seed>: seed to use to generate random initial centroids" << std::endl;
        std::cout << "\t-p: print the data-copy time" << std::endl;
		std::cout << "\t-a: print the total execution time" << std::endl;
        exit(0);
    }

    opts->output_centroids = false;
    opts->print_copy_time  = false;

    struct option l_opts[] = {
        {"k_clusters", required_argument, NULL, 'k'},
        {"n_dims", required_argument, NULL, 'n'},
        {"input", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"output_centroids", no_argument, NULL, 'c'},
        {"seed", required_argument, NULL, 's'},
        {"print_copy_time", no_argument, NULL, 'p'},
	    {"print_total_time", no_argument, NULL, 'a'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:cpa", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->k_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->n_dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (const char *)optarg;
            break;
        case 'm':
            opts->max_num_iters = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c': 
             opts->output_centroids = true;
             break;
        case 'p':
             opts->print_copy_time = true;
             break;
        case 's': 
             opts->seed = atoi((char*)optarg);
             break;
		case 'a':
			 opts->print_total_time = true;
			 break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}