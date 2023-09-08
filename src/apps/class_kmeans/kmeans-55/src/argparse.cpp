#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--n_clusters or -k <n_clusters>" << std::endl;
        std::cout << "\t--dimensions or -d <dimensions>" << std::endl;
        std::cout << "\t--in or -i <file_path>" << std::endl;
        std::cout << "\t--max_iterations or -m <max_iterations>" << std::endl;
        std::cout << "\t--threshold or -t <threshold>" << std::endl;
        std::cout << "\t[Optional] --print-centroids or -c" << std::endl;
        std::cout << "\t--seed or -s" << std::endl;
        std::cout << "\t[Optional] --algorithm or -a (defaults to 0 = sequential)" << std::endl; //TODO
        std::cout << "\t\t 0 = sequential" << std::endl;
        std::cout << "\t\t 1 = thrust" << std::endl;
        std::cout << "\t\t 2 = cuda" << std::endl;
        exit(0);
    }

    opts->print_centroids = false;
    opts->algorithm = 0;

    struct option l_opts[] = {
        {"n_clusters", required_argument, NULL, 'k'},
        {"dimensions", required_argument, NULL, 'd'},
        {"in", required_argument, NULL, 'i'},
        {"max_iterations", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"print-centroids", no_argument, NULL, 'c'},
        {"seed", required_argument, NULL, 's'},
        {"algorithm", optional_argument, NULL, 'a'},
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:a:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->n_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->dimensions = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_iterations = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->print_centroids = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'a':
            opts->algorithm = atoi((char *)optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
