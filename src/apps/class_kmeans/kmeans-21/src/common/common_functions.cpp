#include "common_functions.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

void get_opts(int argc,
              char **argv,
              struct options_t *opts) {
    if (argc == 1) {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_clusters or -k <num_clusters>" << std::endl;
        std::cout << "\t--dims or -d <dimension_of_points>" << std::endl;
        std::cout << "\t--in or -i <input_file_name>" << std::endl;
        std::cout << "\t--max_iters or -m <max_num_of_iterations>" << std::endl;
        std::cout << "\t--threshold or -t <threshold_for_convergence>" << std::endl;
        std::cout << "\t --seed or -s <seed>" << std::endl;
        std::cout << "\t[Optional] --centroids or -c>" << std::endl;
        exit(0);
    }

    opts->output_centroids = false;

    struct option l_opts[] = {
            {"num_clusters", required_argument, NULL, 'k'},
            {"dims",         required_argument, NULL, 'd'},
            {"in",           required_argument, NULL, 'i'},
            {"max_iters",    required_argument, NULL, 'm'},
            {"threshold",    required_argument, NULL, 't'},
            {"seed",         required_argument, NULL, 's'},
            {"centroids",    no_argument,       NULL, 'c'},
            {"shared_memory",optional_argument, NULL, 'o'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:co::", l_opts, &ind)) != -1) {
        switch (c) {
            case 0:
                break;
            case 'k':
                opts->num_cluster = atoi((char *) optarg);
                break;
            case 'd':
                opts->dims = atoi((char *) optarg);
                break;
            case 'i':
                opts->input_file_name = (char *) optarg;
                break;
            case 'm':
                opts->max_num_iter = atoi((char *) optarg);
                break;
            case 't':
                opts->threshold = atof((char *) optarg);
                break;
            case 's':
                opts->seed = atoi((char *) optarg);
                break;
            case 'c':
                opts->output_centroids = true;
                break;
            case 'o':
                if(optarg){
                    opts->shared_memory = atoi((char *) optarg);
                } else {
                    opts->shared_memory = 0;
                }
                break;
            case ':':
                std::cerr << argv[0] << ": option -" << (char) optopt << "requires an argument." << std::endl;
                exit(1);
        }
    }
}

void read_file(struct options_t *args, int *n_vals, float **points) {

    // Open file
    std::ifstream in;
    in.open(args->input_file_name);
    // Get num vals
    in >> *n_vals;
    // Alloc input and output arrays
    *points = (float *) malloc((*n_vals) * args->dims * sizeof(float));
//    for(int i=0; i<*n_vals ; i++){
//        (*points)[i] = (float *) malloc(args->dims * sizeof(float));
//    }
    // Read input vals
    int id;
    for (int i = 0; i < *n_vals; ++i) {
        in >> id;
        for (int j = 0; j < args->dims; ++j) {
            in >> (*points)[i * args->dims + j];
        }

    }
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int) (next / 65536) % (kmeans_rmax + 1);
}

void get_initial_centroids(options_t *opts, int *n_nums, float *points, float **centroids) {

    *centroids = (float *) malloc(opts->num_cluster * opts->dims * sizeof(float));
//    for (int i = 0; i < opts->num_cluster; i++) {
//        (*centroids)[i] = (float *) malloc(opts->dims * sizeof(float));
//    }
    kmeans_srand(opts->seed);
    for (int i = 0; i < opts->num_cluster; ++i) {
        int index = kmeans_rand() % (*n_nums);
        for (int j = 0; j < opts->dims; j++) {
            (*centroids)[i * opts->dims + j] = points[index * opts->dims + j];
        }
    }
}

void recompute_centroids(options_t *opts, int *cluster_mappings, float *points, float *centroids, int *n_vals) {
    int centroid_count[opts->num_cluster];
    for (int i = 0; i < opts->num_cluster; ++i) {
        centroid_count[i] = 0;
        for (int j = 0; j < opts->dims; ++j) {
            centroids[i * opts->dims + j] = 0.0;
        }
    }

    for (int i = 0; i < *n_vals; ++i) {
        centroid_count[cluster_mappings[i]]++;
        for (int j = 0; j < opts->dims; ++j) {
            centroids[cluster_mappings[i] * opts->dims + j] += points[i * opts->dims + j];
        }
    }

    for (int i = 0; i < opts->num_cluster; ++i) {
        for (int j = 0; j < opts->dims; ++j) {
            centroids[i * opts->dims + j] /= (float) centroid_count[i];
        }
    }
}

void assign_clusters(options_t *opts, float *points, float *centroids, int *n_vals, int *cluster_mappings) {
    for (int i = 0; i < *n_vals; ++i) {
        cluster_mappings[i] = assign_cluster(opts, points, i, centroids);
    }
}

int assign_cluster(options_t *opts, float *point, int point_idx, float *centroids) {
    if (opts->num_cluster == 1) {
        return 0;
    }
    int clusterId = 0;
    float min_distance = get_squared_distance(opts->dims, point, point_idx * opts->dims, centroids, 0);
    for (int i = 1; i < opts->num_cluster; ++i) {
        float distance = get_squared_distance(opts->dims, point, point_idx * opts->dims, centroids, i * opts->dims);
        if (min_distance > distance) {
            min_distance = distance;
            clusterId = i;
        }
    }
    return clusterId;
}

float get_squared_distance(int dims, float *point1, int s1, float *point2, int s2) {
    float distance = 0.0;
    for (int i = 0; i < dims; ++i) {
        float temp = point1[s1 + i] - point2[s2 + i];
        distance += temp*temp;
    }
    return distance;
}

void print_cluster_centroids(options_t *opts, float *centroids) {
    for (int clusterId = 0; clusterId < opts->num_cluster; clusterId++) {
        printf("%d", clusterId);
        for (int d = 0; d < opts->dims; d++)
            printf(" %lf", centroids[clusterId * opts->dims + d]);
        printf("\n");
    }
}

void print_cluster_mappings(int *n_vals, int *cluster_mapping) {
    printf("clusters:");
    for (int p = 0; p < *n_vals; p++)
        printf(" %d", cluster_mapping[p]);
    printf("\n");
}

void swap_centroids(float **old_centroids, float **centroids) {
    float *temp = *centroids;
    *centroids = *old_centroids;
    *old_centroids = temp;
}

bool test_convergence(options_t *opts, float *old_centroids, float *centroids) {
    float squared_threshold = opts->threshold * opts->threshold;
    for (int i = 0; i < opts->num_cluster; ++i) {
        float squared_distance = get_squared_distance(opts->dims, old_centroids, i * opts->dims, centroids,
                                                      i * opts->dims);
        if (squared_distance > squared_threshold) {
            return false;
        }
    }
    return true;
}

bool test_convergence(options_t *opts, float *centroid_distances) {
    float squared_threshold = opts->threshold * opts->threshold;
    for (int i = 0; i < opts->num_cluster; ++i) {
        if (centroid_distances[i] > squared_threshold) {
            return false;
        }
    }
    return true;
}