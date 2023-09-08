#include <argparse.h>

void get_opts(int argc,
              char **argv,
              struct kmean_t *kmean)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-k <num_cluster>" << std::endl;
        std::cout << "\t-d <dimension or number of features>" << std::endl;
        std::cout << "\t-m <max_num_iteration>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t-s <seed>" << std::endl;   
        std::cout << "\t-r <revision> (1) Sequential (2) Thrust (3) Cuda Kernel" << std::endl;
        std::cout << "\t[optional]-c <if specified produce centroids>" << std::endl;

        exit(0);
    }

    // Default Params
    kmean->b_centroid = false;
    kmean->n_dims =1;
    kmean->n_max_iter = 150;
    kmean->f_thresh = 1e-10;

    struct option l_opts[] = {
        {"", required_argument, NULL, 'i'},
        {"", required_argument, NULL, 'k'},
        {"", required_argument, NULL, 'd'},
        {"", required_argument, NULL, 'm'},
        {"", required_argument, NULL, 't'},
        {"", required_argument, NULL, 's'},    
        {"", required_argument, NULL, 'r'}, 
        {"", no_argument, NULL, 'c'},

    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "i:k:d:m:t:s:r:c", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            kmean->in_file = (char *)optarg;
                //std::cout << kmean->in_file << std::endl;
            break;
        case 'k':
            kmean->n_clusters = atoi((char *)optarg);
            //std::cout << "Num Clusters = " << kmean->n_clusters << std::endl;
            break;
        case 'd':
            kmean->n_dims = atoi((char *)optarg);
            //std::cout << "Dimension=" << kmean->n_dims << std::endl;
            break;
        case 'm':
            kmean->n_max_iter = atoi((char *)optarg);
            //std::cout << "Max iter=" << kmean->n_max_iter << std::endl;
            break;
        case 't':
            kmean->f_thresh = atof((char *)optarg);
            //std::cout << "Threshold=" << kmean->f_thresh << std::endl;
            break;       
        case 'c':
            kmean->b_centroid = true;
            //std::cout << "Centroid=" << kmean->b_centroid << std::endl;
            break;         
        case 's':
            kmean->n_seed = atoi((char *)optarg);
            //std::cout << "Seed=" << kmean->n_seed << std::endl;
            break;         
        case 'r':
            kmean->n_rev = atoi((char *)optarg);
                if (kmean->n_rev == 1) {
                    //std::cout << "Case 1 : Sequential" << std::endl;
                } else if (kmean->n_rev == 2) { 
                    //std::cout << "Case 2 : Thrust" << std::endl;
                } else if (kmean->n_rev == 3) { 
                    //std::cout << "Case 3 : CUDA Kernel" << std::endl;
                }  else if (kmean->n_rev == 4) { 
                    //std::cout << "Case 4 : CUDA Kernel with Shared Memory" << std::endl;
                } else {
                    std::cerr << "Invalid Program Revision " << std::endl;
                    exit(1);
                }
             break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
    
     
}
