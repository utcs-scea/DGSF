#include "seed.h"
#include <cstring>
#include <cstdlib>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int k_means_rand() {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void k_means_srand(unsigned int seed) {
  next = seed;
}

void k_means_init_random_centroids(int n_points, int d, real *points,
    int k, real *centroids, int seed) {
  k_means_srand(seed); // cmd_seed is a cmdline arg

  for (int i = 0; i < k; i++) {
    int index = k_means_rand() % n_points;
    std::memcpy(&centroids[i * d], &points[index * d], d * sizeof(real));
  }
}
