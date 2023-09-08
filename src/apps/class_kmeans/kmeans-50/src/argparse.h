#include <getopt.h>

struct options_t {
    int num_cluster;
    int dims;
    char *in_file;
    int max_num_iter;
    double threshold;
    int seed;
    int use_shared_mem;
    bool print_centroid;
};


void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t -k <num_cluster>" << std::endl;
        std::cout << "\t -d <dims>" << std::endl;
        std::cout << "\t -i <file_path>" << std::endl;
        std::cout << "\t -m <max_num_iter>" << std::endl;
        std::cout << "\t -t <threshold>" << std::endl;
        std::cout << "\t -s <seed>" << std::endl;
        std::cout << "\t -h <use_shared_mem>" << std::endl;
        std::cout << "\t[Optional] -c" << std::endl;
        exit(0);
    }

    opts->print_centroid = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"file_path", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"use_shared_mem", required_argument, NULL, 'h'},
        {"print_centroid", no_argument, NULL, 'c'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:h:c", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            //std::cout << "num_cluster:" << opts->num_cluster << std::endl;
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            //std::cout << "dims:" << opts->dims << std::endl;
            break;        
        case 'i':
            opts->in_file = (char *)optarg;
            //std::cout << "in_file:" << opts->in_file << std::endl;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            //std::cout << "max_num_iter:" << opts->max_num_iter << std::endl;
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            //std::cout << "threshold:" << opts->threshold << std::endl;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            //std::cout << "seed:" << opts->seed << std::endl;
            break;
        case 'c':
            opts->print_centroid = true;
            //std::cout << "print_centroid:" << opts->print_centroid << std::endl;
            break;
        case 'h':
            opts->use_shared_mem = atoi((char *)optarg);
            //std::cout << "use_shared_mem:" << opts->use_shared_mem << std::endl;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }

}
