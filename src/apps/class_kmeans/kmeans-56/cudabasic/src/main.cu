#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// Copied from the NVIDIA_SDK_Samples code that comes with CUDA
// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

struct options_t {
  int n_clusters;
  int n_dims;
  char *in_file;
  int n_iter;
  double threshold;
  bool output_centroids;
  int seed;
};

void get_opts(int argc, char **argv, struct options_t *opts);
template <typename Vector1, typename Vector2>
void dense_histogram(const Vector1 &input, Vector2 &histogram);

struct output_t {
  double **result_centers;      // this is the list of centroids
  double **input_vals;          // this is the list of points (inputs)
  int **point_centroid_id_map;  // this is a mapping from the point's index to
                                // its centroid
  int *n_iter_to_converge;
  double *time_per_iter_ms;
};

void read_file(
    struct options_t *args, int *n_vals,
    double **input_vals,  // inputs here is a pointer to  1d array representing
                          // the list of n_vals values of d dimenesions
    double *
        *result_centers,  // centers is a pointer to the 1d array representing
                          // the list of k centroids of d dimensions
    int **point_centers_map  // this is 1D array ie a map from point id to
                             // center id dimension is 1xN where each index has
                             // the value of the index of the cluster to which a
                             // point of the index belongs to
);

void write_result(struct output_t *info, struct options_t *args, int *n_vals);

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand();
void kmeans_srand(unsigned int);
void print_centroids(struct output_t *info, struct options_t *args);
void print_point(double *arr, int num);
void print_labels(int *l, int n);

void compute_means(int n_vals, options_t *args, output_t *output);
__global__ void find_nearest_centroids(double *points, double *centroids, int k,
                                       int dim, int n, int *labels);
__device__ double euclidean_distance(double *point1, int offset_one,
                                     double *point2, int offset2, int dim);
double euclidean_distance_host(double **point1, int offset_one, double **point2,
                               int offset_two, int dim);

void average_labeled_centroids(double **points, int **labels, int k, int dim,
                               int n, double **centroids);
__global__ void converged(double *centroids, double *old_centroids,
                          bool *s_converged, double threshold, int k, int dim,
                          bool *is_converged);

int main(int argc, char **argv) {
  // Parse args
  struct options_t opts;
  get_opts(argc, argv, &opts);

  int n_vals;
  double *inputs = NULL;
  double *centers = NULL;
  int *point_center_map = NULL;

  read_file(&opts, &n_vals, &inputs, &centers, &point_center_map);

  struct output_t output;

  output.input_vals = &inputs;
  output.result_centers = &centers;
  output.point_centroid_id_map = &point_center_map;
  output.n_iter_to_converge = NULL;
  output.time_per_iter_ms = NULL;

  compute_means(n_vals, &opts, &output);

  write_result(&output, &opts, &n_vals);

  // free memory
  free(inputs);
  free(centers);
  free(point_center_map);
  return 0;
}

/// Argparse

void get_opts(int argc, char **argv, struct options_t *opts) {
  if (argc == 1) {
    std::cout << "Usage:" << std::endl;

    std::cout << "\t--k or -k <num_cluster>" << std::endl;
    std::cout << "\t--d or -d <dims>" << std::endl;
    std::cout << "\t--in or -i <file_path>" << std::endl;
    std::cout << "\t--m or -m <max_num_iter>" << std::endl;
    std::cout << "\t--t or -t <threshold>" << std::endl;
    std::cout << "\t--c or -c" << std::endl;
    std::cout << "\t --s or -s <seed>" << std::endl;
    exit(0);
  }

  opts->output_centroids = false;

  struct option l_opts[] = {
      {"k", required_argument, NULL, 'k'},  {"d", required_argument, NULL, 'd'},
      {"in", required_argument, NULL, 'i'}, {"m", required_argument, NULL, 'm'},
      {"t", required_argument, NULL, 't'},  {"s", required_argument, NULL, 's'},
      {"c", no_argument, NULL, 'c'}};

  int ind, c;
  while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:c", l_opts, &ind)) != -1) {
    switch (c) {
      case 0:
        break;
      case 'k':
        opts->n_clusters = atoi((char *)optarg);
        break;
      case 'd':
        opts->n_dims = atoi((char *)optarg);
        break;
      case 'i':
        opts->in_file = (char *)optarg;
        break;
      case 'm':
        opts->n_iter = atoi((char *)optarg);
        break;
      case 't':
        opts->threshold = atof((char *)optarg);
        break;
      case 's':
        opts->seed = atoi((char *)optarg);
        break;
      case 'c':
        opts->output_centroids = true;
        break;
      case ':':
        std::cerr << argv[0] << ": option -" << (char)optopt
                  << "requires an argument." << std::endl;
        exit(1);
    }
  }
}

// CPUIO

void read_file(struct options_t *args, int *n_vals, double **input_vals,
               double **result_centers, int **point_center_map) {
  // Open file
  std::ifstream in;
  in.open(args->in_file);

  // Get num points
  in >> *n_vals;

  // Alloc input and output arrays
  *input_vals = (double *)malloc(sizeof(double) * (*n_vals * args->n_dims));

  *result_centers =
      (double *)malloc(sizeof(double) * args->n_clusters * args->n_dims);

  *point_center_map = (int *)malloc(sizeof(int) * *n_vals);
  for (int i = 0; i < *n_vals; i++) {
    (*point_center_map)[i] = -1;  // initialize
  }

  // Read input vals
  for (int i = 0; i < *n_vals; ++i) {
    char *raw_string = (char *)malloc(500 * sizeof(char));
    auto offset = i * args->n_dims;
    for (int j = 0; j < args->n_dims + 1; j++) {
      in >> raw_string;
      if (j != 0) {
        (*input_vals)[offset + (j - 1)] = atof(raw_string);
      }
    }
    free(raw_string);
  }
}

void write_result(struct output_t *info, struct options_t *args, int *n_vals) {
  printf("%d,%lf\n", *info->n_iter_to_converge, *info->time_per_iter_ms);
  if (!args->output_centroids) {
    printf("clusters:");
    for (int p = 0; p < *n_vals; p++) {
      int cluster_id = info->point_centroid_id_map[0][p];
      printf(" %d", cluster_id);
    }
  } else {
    for (int cluster_id = 0; cluster_id < args->n_clusters; cluster_id++) {
      printf("%d ", cluster_id);
      auto offset = cluster_id * args->n_dims;
      for (int d = 0; d < args->n_dims; d++) {
        double *centers = *(info->result_centers);
        printf("%lf ", centers[offset + d]);
      }
      printf("\n");
    }
  }
}

// K means

void compute_means(int n_vals, options_t *args, output_t *output) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int dim = args->n_dims;
  int k = args->n_clusters;
  double **centroids = (output->result_centers);
  double **points = (output->input_vals);
  int **labels = (output->point_centroid_id_map);

  int n = n_vals;
  double *d_centroids = NULL;
  double *d_old_centroids = NULL;
  double *d_points = NULL;
  int *d_labels = NULL;
  bool *d_is_converged = NULL;
  bool *d_s_converged = NULL;

  checkCudaErrors(cudaMalloc(&d_centroids, k * dim * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_old_centroids, k * dim * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_points, n * dim * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_labels, n * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_is_converged, sizeof(bool)));
  checkCudaErrors(
      cudaMalloc(&d_s_converged,
                 2048 * sizeof(bool)));  // MAX being the number of clusters

  // Start the timer
  cudaEventRecord(start);

  // Assign random centers

  int cmd_seed = args->seed;
  kmeans_srand(cmd_seed);
  int num_iteration = 0;

  // randomly assign centroids
  for (int i = 0; i < k; i++) {
    int index = kmeans_rand() % n_vals;
    int point_offset = index * dim;
    int centroid_offset = i * dim;
    for (int j = 0; j < dim; j++) {
      (*centroids)[centroid_offset + j] = (*points)[point_offset + j];
    }
  }

  // Now write a custom map/reduce style code
  bool done = false;
  bool is_converged = true;
  // TODO: make this work with device memory

  checkCudaErrors(cudaMemcpy(d_centroids, *centroids, k * dim * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_points, *points, n * dim * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_labels, *labels, n * sizeof(int), cudaMemcpyHostToDevice));

  while (!done) {
    is_converged = true;
    checkCudaErrors(cudaMemcpy(d_old_centroids, *centroids,
                               k * dim * sizeof(double),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_is_converged, &is_converged, sizeof(bool),
                               cudaMemcpyHostToDevice));

    num_iteration++;
    // printf("before\n");
    // print_labels(*labels, n);
    // printf("centroids\n");
    // print_point(*centroids, dim * k);
    find_nearest_centroids<<<n_vals, 1>>>(d_points, d_centroids, k, dim, n_vals,
                                          d_labels);
    cudaDeviceSynchronize();
    // copy data back
    cudaMemcpy(*centroids, d_centroids, k * dim * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(*labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("after\n");
    // print_labels(*labels, n);
    // reset

    average_labeled_centroids(points, labels, k, dim, n_vals, centroids);
    // printf("computed centroids\n");
    // print_point(*centroids, dim * k);
    // copy labels to device
    cudaMemcpy(d_labels, *labels, n * sizeof(int), cudaMemcpyHostToDevice);

    // copy new centroids to device

    checkCudaErrors(cudaMemcpy(d_centroids, *centroids,
                               k * dim * sizeof(double),
                               cudaMemcpyHostToDevice));

    // std::cout << centroids[0].size() << std::endl;
    int th = args->threshold;
    converged<<<k, 1>>>(d_centroids, d_old_centroids, d_s_converged, th, k, dim,
                        d_is_converged);
    checkCudaErrors(cudaMemcpy(&is_converged, d_is_converged, sizeof(bool),
                               cudaMemcpyDeviceToHost));

    // std::cout << "converge: " << is_conv << "\n";
    done = (num_iteration > args->n_iter) || is_converged;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float diff = 0;
  cudaEventElapsedTime(&diff, start, stop);

  double time_per_iteration = (num_iteration * 1.0) / diff;
  output->time_per_iter_ms = &time_per_iteration;
  output->n_iter_to_converge = &num_iteration;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Performs parallel reduction using CUDA shared memory
__global__ void converged(double *centroids, double *old_centroids,
                          bool *s_converged, double threshold, int k, int dim,
                          bool *is_converged) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > k) {
    return;
  }

  //__shared__ bool s_converged[2048];  // take max
  s_converged[idx] = true;

  __syncthreads();

  auto centroid_offset = idx * dim;
  auto distance = euclidean_distance(centroids, centroid_offset, old_centroids,
                                     centroid_offset, dim);
  if (distance > threshold) {
    s_converged[idx] = false;
    if (idx != 0) {
      return;
    }
  }

  __syncthreads();
  if (idx == 0) {
    for (auto i = 0; i < k; i++) {
      *is_converged = (*is_converged) && s_converged[i];
      if (!(*is_converged)) {
        return;
      }
    }
  }

  return;
}

void average_labeled_centroids(double **points, int **labels, int k, int dim,
                               int n, double **centroids) {
  int index_count_map[k];
  double new_centroids[k * dim];

  for (int i = 0; i < k * dim; i++) {
    new_centroids[i] = 0;
  }

  for (int i = 0; i < k; i++) {
    index_count_map[i] = 0;
  }

  // take sum of all the centroids that map to the same label i
  // also keep track of how many centroids map to the same label i
  for (int i = 0; i < n; i++) {
    auto centroid_id = (*labels)[i];

    auto centroid_offset = centroid_id * dim;
    auto offset = i * dim;
    for (auto j = 0; j < dim; j++) {
      new_centroids[centroid_offset + j] += (*points)[offset + j];
    }
    index_count_map[centroid_id] += 1;
  }

  // divide by the total number of centroids that mapped to same label i
  for (int i = 0; i < k; i++) {
    auto offset = i * dim;
    for (int j = 0; j < dim; j++) {
      (*centroids)[offset + j] = new_centroids[offset + j] / index_count_map[i];
    }
  }
  return;
}

__global__ void find_nearest_centroids(double *points, double *centroids, int k,
                                       int dim, int n, int *labels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) {
    return;
  }
  auto offset_point = i * dim;
  auto minimum = INFINITY;
  auto index = -1;
  for (auto j = 0; j < k; j++) {
    auto centroid_offset = j * dim;
    // TODO: debug this. this is failng
    auto distance = euclidean_distance(points, offset_point, centroids,
                                       centroid_offset, dim);
    if (distance < minimum) {
      minimum = distance;
      index = j;
    }
  }
  // printf("label of i is %d %d %d\n", i, index, minimum);
  labels[i] = index;
}

// returns the euclidean distance between 2 dim dimensionals points point1 and
// point 2
__device__ double euclidean_distance(double *point1, int offset_one,
                                     double *point2, int offset_two, int dim) {
  double sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += (point1[offset_one + i] - point2[offset_two + i]) *
           (point1[offset_one + i] - point2[offset_two + i]);
  }
  return sqrt(sum);
}

double euclidean_distance_host(double **point1, int offset_one, double **point2,
                               int offset_two, int dim) {
  double sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += ((*point1)[offset_one + i] - (*point2)[offset_two + i]) *
           ((*point1)[offset_one + i] - (*point2)[offset_two + i]);
  }
  return sqrt(sum);
}

// UTIL

int kmeans_rand() {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}
void kmeans_srand(unsigned int seed) { next = seed; }

void print_centroids(struct output_t *output, struct options_t *args) {
  printf("Current centroids\n");
  for (int i = 0; i < args->n_clusters; i++) {
    auto offset = i * args->n_dims;
    for (int j = 0; j < args->n_dims; j++) {
      printf("%f ", (*output->result_centers[offset + j]));
    }
    printf("\n");
  }
}

void print_point(double *arr, int num) {
  for (int j = 0; j < num; j++) {
    printf("%f ", arr[j]);
  }
  printf("\n");
}

void print_labels(int *arr, int num) {
  for (int j = 0; j < num; j++) {
    printf("%d->%d\n", j, arr[j]);
  }
  printf("--------\n");
}
