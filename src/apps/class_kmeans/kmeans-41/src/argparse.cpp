#include "argparse.h"

void get_opts(int argc, char **argv, struct options_t *opts) {
    if (argc == 1) {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-k <num_cluster>" << std::endl;
        std::cout << "\t-d <dimensions>" << std::endl;
        std::cout << "\t-m <max_iteration>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t-s <random_seed>" << std::endl;
        std::cout << "\t[Optional] -c to print centroids" << std::endl;
        std::cout << "\t[Optional] -x to use cuda shared memory" << std::endl;
        std::cout << "\t[Optional] -e to print end to end CUDA runtime duration" << std::endl;
        exit(0);
    }

    opts->output_centroids = false;
    opts->cuda_shared = false;
    opts->print_e2e = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"output_centroids", no_argument, NULL, 'c'},
        {"cuda_shared", no_argument, NULL, 'x'},
        {"print_e2e", no_argument, NULL, 'e'}
    };
    
    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:xe", l_opts, &ind)) != -1) {
        switch (c) {
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
                opts->threshold = atof((char*) optarg);
                break;
            case 'c':
                opts->output_centroids = true;
                break;
            case 's':
                opts->seed = atoi((char*) optarg);
                break;
            case 'x':
                opts->cuda_shared = true;
                break;
            case 'e':
                opts->print_e2e = true;
                break;
            case ':':
                std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
                exit(1);
        }
    }
}