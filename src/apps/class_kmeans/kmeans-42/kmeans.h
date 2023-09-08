

struct options_t {
    int num_cluster;
    int dims;
    char *inputfilename;
    int max_num_iter;
    double threshold;
    bool c;
    int seed;
    int type;
    int num_threads;
};


int sequential_kmeans(int n_vals, int dims, int num_centroids, double** starting_centroids, double *input_vals, double threshold, int max_num_iter, bool c);

double euclidean_distance(double* x, double* y, int size);

void parallel_thrust_kmeans(int n_vals, int dims, int num_centroids, double* centroids, double *input_vals, double threshold, int max_num_iter, bool c);

void print_arr(double* centroids, int dims, int num_cluster);

void cuda_basic_kmeans(int n_vals, int dims, int num_centroids, double *centroids, double *input_vals, double threshold, int max_num_iter, bool c, int threadsPerBlock);

void cuda_shmem_kmeans(int n_vals, int dims, int num_centroids, double *centroids, double *input_vals, double threshold, int max_num_iter, bool c, int threadsPerBlock);
