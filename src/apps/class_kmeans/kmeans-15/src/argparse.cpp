#include <argparse.h>

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--in or -i <file_path>" << std::endl;
        std::cout << "\t--n_cluster or -k <num_cluster>" << std::endl;
        std::cout << "\t--n_dims or -d <num_dimension>" << std::endl;
        std::cout << "\t--max_iter or -m <max_interations>" << std::endl;
        std::cout << "\t--threshold or -t <threshold_value>" << std::endl;
        std::cout << "\t--seed or -s <seed_value>" << std::endl;
        std::cout << "\t[Optional] --control or -c" << std::endl;
        exit(0);
    }

    opts->control = false;

    struct option l_opts[] = {
        {"in", required_argument, NULL, 'i'},
        {"n_cluster", required_argument, NULL, 'k'},
        {"n_dims", required_argument, NULL, 'd'},
        {"max_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"control", no_argument, NULL, 'c'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "i:k:d:m:t:s:c", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'k':
            opts->n_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->n_dims = atoi((char *)optarg);
            break;
        case 'm':
            opts->max_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 's':
            opts->seed = std::stoul((char *)optarg, NULL, 10);
            break;
        case 'c':
            opts->control = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
