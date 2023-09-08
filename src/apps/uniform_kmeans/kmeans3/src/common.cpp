#include <fstream>
#include <iostream>

#include "common.h"


static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

struct option l_opts[] = {
    {"num_cluster ", required_argument, NULL, 'k'},
    {"dims ", required_argument, NULL, 'd'},
    {"inputfilename", required_argument, NULL, 'i'},
    {"max_num_iter", required_argument, NULL, 'm'},
    {"threshold", required_argument, NULL, 't'},
    {"c", no_argument, NULL, 'c'},
    {"seed", required_argument, NULL, 's'},
};

int kmeans_rand()
{
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed)
{
    next = seed;
}

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1) {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <num_cluster>\n";
        std::cout << "\t--dims or -d <dims>\n";
        std::cout << "\t--inputfilename or -i <in_file>\n";
        std::cout << "\t--max_num_iter or -m <max_num_iter>\n";
        std::cout << "\t--threshold or -t <threshold>\n";
        std::cout << "\t[Optional] -c\n";
        std::cout << "\t--seed or -s <seed>\n";

        exit(0);
    }

    opts->output_centroid = false;

    int option;
    while ((option = getopt(argc, argv, "k:d:i:m:t:cs:")) != -1) {
        switch (option) {
            case 0:
                break;
            case 'k':
                opts->num_cluster = atoi((char *)optarg);
                break;
            case 'd':
                opts->dims = atoi((char *)optarg);
                break;
            case 'i':
                opts->in_file = (char *)optarg;
                break;
            case 'm':
                opts->max_num_iter = atoi((char *)optarg);
                break;
            case 't':
                opts->threshold = atof((char *)optarg);
                break;
            case 'c':
                opts->output_centroid = true;
                break;    
            case 's':
                opts->seed = atoi((char *)optarg);
                break;
            case ':':
                std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
                exit(1);
        }
    }    
}

void read_file(const struct options_t &opts,
               int &n_points,
               std::vector<float> &points)
{
  	// Open file
	std::ifstream in;
	in.open(opts.in_file);

    if (!in.is_open()) {
        std::cerr << "Error opening file.\n";
        throw "Error opening file.\n";
    }
    
	// Get num points
	in >> n_points;
    points = std::vector<float>(n_points * opts.dims);

	// Read input vals
	for (int i = 0; i < n_points; ++i) {
        float value;
        in >> value; // skip row id
        for (int j = 0; j < opts.dims; ++j) {
            in >> value;
            points[i * opts.dims + j] = value;
        }    
	}
}

void generate_centroids(
    const std::vector<float> &points,
    std::vector<float> &centroids,
    const struct options_t &opts)
{
    kmeans_srand(opts.seed);
    for (int i=0; i < opts.num_cluster; ++i) {
        int index = kmeans_rand() % (points.size() / opts.dims);

        for (int j = 0; j < opts.dims; ++j) {
            centroids[i * opts.dims + j] = points[index * opts.dims + j];
        }
    }
}

void output_results(
    const std::vector<int> &labels,
    const std::vector<float> &centroids,
    float total_time_in_ms,
    int iter_to_converge,
    const struct options_t &opts)
{
    printf(
        "%d,%lf\n",
        iter_to_converge,
        total_time_in_ms / iter_to_converge);

    if (opts.output_centroid) {
        for (int i = 0; i < opts.num_cluster; ++i) {
            printf("%d ", i);
            for (int d = 0; d < opts.dims; ++d) {
                printf("%lf ", centroids[i * opts.dims + d]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (unsigned int p = 0; p < labels.size(); ++p) {
            printf(" %d", labels[p]);
        }
    }
}