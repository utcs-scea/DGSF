#include <chrono>
#include <cfloat>
#include "argparse.h"
#include "io.h"
#include "common.h"
#include "seed.h"
#include "k_means_sequential.h"
#include "k_means_thrust.h"
#include "k_means_cuda.h"

int main(int argc, char **argv) {
  // Parse args
  struct options_t opts;
  get_opts(argc, argv, &opts);

  int n_points;
  real *points;
  read_file(&opts, &n_points, &points);

  int *point_cluster_ids = (int *)malloc(n_points * sizeof(int));
  real *centroids = (real *)malloc(opts.n_clusters * opts.dimensions * sizeof(real));
  k_means_init_random_centroids(n_points, opts.dimensions, points,
      opts.n_clusters, centroids, opts.seed);

  int iterations = 0;
  double per_iteration_time = 0;

  // Start timer
  auto start = std::chrono::high_resolution_clock::now();

  switch (opts.algorithm)
  {
    case 0:
      DEBUG_OUT("Running k_means_sequential:");

      iterations = k_means_sequential(n_points, points, &opts, point_cluster_ids, &centroids);

      DEBUG_OUT("Finished k_means_sequential:");
      break;
    case 1:
      DEBUG_OUT("Running k_means_thrust:");

      iterations = k_means_thrust(n_points, points, &opts, point_cluster_ids, &centroids, &per_iteration_time);

      DEBUG_OUT("Finished k_means_thrust:");
      break;
    case 2:
      DEBUG_OUT("Running k_means_cuda:");

      iterations = k_means_cuda(n_points, points, &opts, point_cluster_ids, &centroids, &per_iteration_time);

      DEBUG_OUT("Finished k_means_cuda:");
      break;
    case 3:
      DEBUG_OUT("Running k_means_cuda shmem:");

      iterations = k_means_cuda(n_points, points, &opts, point_cluster_ids, &centroids, &per_iteration_time);

      DEBUG_OUT("Finished k_means_cuda shmem:");
      break;
  }

  //End timer and print out elapsed
  auto end = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration<double, std::milli>(end - start);

  if (opts.algorithm == 0) {
    per_iteration_time = diff.count() / iterations;
  }
  else {
    TIMING_PRINT(printf("Per iteration chrono: %f ms \n", diff.count() / iterations));
  }

  printf("%d,%lf\n", iterations, per_iteration_time);

  if (opts.print_centroids) {
    PRINT_CENTROIDS(centroids, opts.dimensions, opts.n_clusters);
  }
  else {
    printf("clusters:");
    for (int i = 0; i < n_points; i++)
      printf(" %d", point_cluster_ids[i]);
  }

  free(centroids);
  free(point_cluster_ids);
  free(points);
}
