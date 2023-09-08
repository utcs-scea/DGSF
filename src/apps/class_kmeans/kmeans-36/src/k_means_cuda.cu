#include <iostream>

#include "k_means_cuda.h"
#include "common.h"

#include <cuda_runtime.h>
#include "ext/helper_cuda.h"

__device__ double ratomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void compute_point_centroid_distances_shmem(int d, int k, int n_points,
    real *d_points, real *centroids, real *point_centroid_distances) {
  const unsigned int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int centroid_id = threadIdx.y;
  const unsigned int i = threadIdx.z;

  extern __shared__ real s[];
  real *s_centroids = s;
  real *sums = s + d * blockDim.y;

  if (point_id >= n_points) {
    return ;
  }

  s_centroids[centroid_id*d + i] = centroids[centroid_id*d + i];
  sums[threadIdx.x*k + centroid_id] = 0;

  __syncthreads();

  real p = d_points[point_id*d + i];

#ifdef REAL_FLOAT
  atomicAdd(sums + threadIdx.x*k + centroid_id, POW2(p - s_centroids[centroid_id*d + i]));
#else
  ratomicAdd(sums + threadIdx.x*k + centroid_id, POW2(p - s_centroids[centroid_id*d + i]));
#endif

    // if (point_id == 8) {
    //   DEBUG_PRINT(printf("i %d p %f c %f s %f", point_id, p, c, sum));
    // }
  __syncthreads();

  point_centroid_distances[point_id*k + centroid_id] = sums[threadIdx.x*k + centroid_id];
  // if (point_id == 8) {
  //   DEBUG_PRINT(printf("p %d k %d d %f ", point_id, centroid_id, sum));
  // }
}

__global__ void compute_point_centroid_distances(int d, int k, int n_points,
    real *d_points, real *centroids, real *point_centroid_distances) {
  const unsigned int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int centroid_id = threadIdx.y;

  if (point_id >= n_points) {
    return ;
  }

  real p = d_points[point_id*d + threadIdx.z];
  real c = centroids[centroid_id*d + threadIdx.z];

  // if (point_id == 8) {
  //   DEBUG_PRINT(printf("i %d p %f c %f s %f", point_id, p, c, sum));
  // }
#ifdef REAL_FLOAT
  atomicAdd(point_centroid_distances + point_id*k + centroid_id, POW2(p - c));
#else
  ratomicAdd(point_centroid_distances + point_id*k + centroid_id, POW2(p - c));
#endif
  // if (point_id == 8) {
  //   DEBUG_PRINT(printf("p %d k %d d %f ", point_id, centroid_id, sum));
  // }
}

__global__ void compute_point_centroid_ids(int d, int k, int n_points,
    real *d_points, real *point_centroid_distances,
    unsigned int* d_k_counts, int *d_point_cluster_ids) {
  const unsigned int point_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (point_id >= n_points) {
    return ;
  }

  int nearest_centroid = 0;
  real nearest_centroid_dist = point_centroid_distances[point_id*k + nearest_centroid];

  for (int i = 1; i < k; i++) {
    real next_distance = point_centroid_distances[point_id*k+i];
    if (nearest_centroid_dist > next_distance) {
      nearest_centroid_dist = next_distance;
      nearest_centroid = i;
    }
  }
  d_point_cluster_ids[point_id] = nearest_centroid;

  // if (point_id == 8) {
  //   DEBUG_PRINT(printf("p %d c %d d %f", point_id, nearest_centroid, nearest_centroid_dist));
  // }

  atomicInc(d_k_counts + nearest_centroid, n_points);
}

__global__ void compute_new_centroids_shmem(int d, int n_points, real *d_points,
    unsigned int* d_k_counts, int *d_point_cluster_ids, real *centroids) {
  const unsigned int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int nearest_centroid = d_point_cluster_ids[point_id];
  const int i = threadIdx.y;

  extern __shared__ unsigned int s_new_centroids[];
  unsigned int *s_k_counts = s_new_centroids;
  s_k_counts[nearest_centroid] = d_k_counts[nearest_centroid];

  if (point_id >= n_points) {
    return ;
  }

#ifdef REAL_FLOAT
  atomicAdd(centroids + nearest_centroid*d+i,
      d_points[point_id*d+i]/s_k_counts[nearest_centroid]);
#else
  ratomicAdd(centroids + nearest_centroid*d+i,
      d_points[point_id*d+i]/s_k_counts[nearest_centroid]);
#endif
}

__global__ void compute_new_centroids(int d, int n_points, real *d_points,
    unsigned int* d_k_counts, int *d_point_cluster_ids, real *centroids) {
  const unsigned int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int nearest_centroid = d_point_cluster_ids[point_id];
  const unsigned int count = d_k_counts[nearest_centroid];
  const int i = threadIdx.y;

  if (point_id >= n_points) {
    return ;
  }

#ifdef REAL_FLOAT
  atomicAdd(centroids + nearest_centroid*d+i,
      d_points[point_id*d+i]/count);
#else
  ratomicAdd(centroids + nearest_centroid*d+i,
      d_points[point_id*d+i]/count);
#endif
}

__global__ void compute_converged(int k, int d, real thresh,
    real *new_centroids, real *old_centroids, bool *d_converged) {
  if (threadIdx.x < k*d &&
      abs(new_centroids[threadIdx.x] - old_centroids[threadIdx.x]) >= thresh) {
    *d_converged = false;
  }
}

__global__ void merge_centroids(int k, int d,
    real *new_centroids, real *old_centroids) {
  if (threadIdx.x >= k) {
    return ;
  }

  bool vanished = true;

  for (int i = 0; i < d; i++) {
    vanished = vanished && (new_centroids[threadIdx.x*k + i] == 0);
  }

  if (vanished) {
    for (int i = 0; i < d; i++) {
      new_centroids[threadIdx.x*k + i] = old_centroids[threadIdx.x*k + i];
    }
  }
}

int k_means_cuda(int n_points, real *points, struct options_t *opts,
  int* point_cluster_ids, real** centroids, double *per_iteration_time) {

  bool use_shared_mem = (opts->algorithm == 3);
  int device_id = gpuGetMaxGflopsDeviceId();

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
  checkCudaErrors(cudaSetDevice(device_id));
  DEBUG_PRINT(printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", device_id, deviceProp.name));

  bool done = false;
  int iterations = 0;
  int k = opts->n_clusters;
  int d = opts->dimensions;

  cudaEvent_t start, stop, in_start, in_stop, out_start, out_stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&in_start);
  cudaEventCreate(&in_stop);
  cudaEventCreate(&out_start);
  cudaEventCreate(&out_stop);

  DEBUG_OUT("CUDA malloc");
  cudaEventRecord(start, 0);

  real *d_points;
  checkCudaErrors(cudaMalloc(&d_points, n_points * d * sizeof(real)));

  real *old_centroids;
  checkCudaErrors(cudaMalloc(&old_centroids, k * d * sizeof(real)));

  real *new_centroids;
  checkCudaErrors(cudaMalloc(&new_centroids, k * d * sizeof(real)));

  real *temp_centroids;

  real *point_centroid_distances;
  checkCudaErrors(cudaMalloc(&point_centroid_distances, n_points * k * sizeof(real)));

  int *d_point_centroid_ids;
  checkCudaErrors(cudaMalloc(&d_point_centroid_ids, n_points * sizeof(int)));
  // dv_int unsorted_d_point_cluster_ids(n_points);

  unsigned int *d_k_counts;
  checkCudaErrors(cudaMalloc(&d_k_counts, k * sizeof(int)));

  bool *d_converged;
  checkCudaErrors(cudaMalloc(&d_converged, sizeof(bool)));

  DEBUG_OUT("CUDA copy");
  // timer code - 0_Simple/simpleMultiCopy/simpleMultiCopy.cu
  cudaEventRecord(in_start, 0);
  checkCudaErrors(cudaMemcpyAsync(d_points, points, n_points * d * sizeof(real),
                                  cudaMemcpyHostToDevice, 0));

  checkCudaErrors(cudaMemcpyAsync(old_centroids, *centroids, k * d * sizeof(real),
                                  cudaMemcpyHostToDevice, 0));
  cudaEventRecord(in_stop,0);
  cudaEventSynchronize(in_stop);

  float memcpy_h2d_time;
  cudaEventElapsedTime(&memcpy_h2d_time, in_start, in_stop); //TODO: is async messing it up?
  TIMING_PRINT(printf("Host to device: %f ms \n", memcpy_h2d_time));

  int max_thread_size = 1024;

  while(!done) {
    if (DEBUG_TEST) {
      checkCudaErrors(cudaMemcpy(*centroids, old_centroids, k * d * sizeof(real),
            cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaDeviceSynchronize());
      DEBUG_OUT("Old centroids:");
      PRINT_CENTROIDS(*centroids, d, k);
    }

    dim3 pcd_threads(((max_thread_size) / k )/ d, k, d); //or 4?
    dim3 pcd_grid(n_points / pcd_threads.x + (n_points % pcd_threads.x == 0 ? 0 : 1));

    if (use_shared_mem) {
      unsigned int shmem_size = pcd_threads.y * d * sizeof(real) + pcd_threads.x * pcd_threads.y * sizeof(real);

      DEBUG_PRINT(printf("compute_point_centroid_distances_shmem: b: %d %d %d t: %d %d %d s: %d\n",
            pcd_grid.x, pcd_grid.y, pcd_grid.z, pcd_threads.x, pcd_threads.y, pcd_threads.z, shmem_size));

      compute_point_centroid_distances_shmem<<< pcd_grid, pcd_threads, shmem_size >>>(d, k, n_points,
          d_points, old_centroids, point_centroid_distances);
    }
    else {
      DEBUG_PRINT(printf("compute_point_centroid_distances: b: %d %d %d t: %d %d %d\n",
            pcd_grid.x, pcd_grid.y, pcd_grid.z, pcd_threads.x, pcd_threads.y, pcd_threads.z));

      compute_point_centroid_distances<<< pcd_grid, pcd_threads >>>(d, k, n_points,
          d_points, old_centroids, point_centroid_distances);
    }

    getLastCudaError("compute_point_centroid_distances() execution failed\n");
    // checkCudaErrors(cudaDeviceSynchronize());
    DEBUG_OUT(" ");

    dim3 pci_threads(max_thread_size); // could include k dim
    dim3 pci_grid(n_points / pci_threads.x + (n_points % pci_threads.x == 0 ? 0 : 1));

    DEBUG_PRINT(printf("compute_point_centroid_ids: b: %d %d %d t: %d %d %d\n",
          pci_grid.x, pci_grid.y, pci_grid.z, pci_threads.x, pci_threads.y, pci_threads.z));

    checkCudaErrors(cudaMemset(d_k_counts, 0, k * sizeof(int)));
    checkCudaErrors(cudaMemset(new_centroids, 0, k * d * sizeof(real)));

    compute_point_centroid_ids<<< pci_grid, pci_threads >>>(d, k, n_points, d_points,
        point_centroid_distances, d_k_counts, d_point_centroid_ids);

    getLastCudaError("compute_point_centroid_ids() execution failed\n");
    // checkCudaErrors(cudaDeviceSynchronize());
    DEBUG_OUT(" ");

    dim3 cnc_threads(max_thread_size / d, d);
    dim3 cnc_grid(n_points / cnc_threads.x + (n_points % cnc_threads.x == 0 ? 0 : 1));

    if (use_shared_mem) {
      unsigned int shmem_size = k * sizeof(unsigned int);

      DEBUG_PRINT(printf("compute_new_centroids: b: %d %d %d t: %d %d %d s: %d\n",
            cnc_grid.x, cnc_grid.y, cnc_grid.z, cnc_threads.x, cnc_threads.y, cnc_threads.z, shmem_size));

      compute_new_centroids<<< cnc_grid, cnc_threads, shmem_size >>>(d, n_points, d_points,
          d_k_counts, d_point_centroid_ids, new_centroids);
    }
    else {
      DEBUG_PRINT(printf("compute_new_centroids: b: %d %d %d t: %d %d %d\n",
            cnc_grid.x, cnc_grid.y, cnc_grid.z, cnc_threads.x, cnc_threads.y, cnc_threads.z));

      compute_new_centroids<<< cnc_grid, cnc_threads >>>(d, n_points, d_points,
          d_k_counts, d_point_centroid_ids, new_centroids);
    }

    getLastCudaError("compute_new_centroids() execution failed\n");
    // checkCudaErrors(cudaDeviceSynchronize());
    DEBUG_OUT(" ");

    DEBUG_PRINT(printf("merge_centroids:\n"));

    merge_centroids<<< (k / 512)+1, 512 >>>(k, d, new_centroids, old_centroids);

    getLastCudaError("merge_converged() execution failed\n");
    // checkCudaErrors(cudaDeviceSynchronize());
    DEBUG_OUT(" ");

    // swap centroids
    temp_centroids = new_centroids;
    new_centroids = old_centroids;
    old_centroids = temp_centroids;

    real l1_thresh = opts->threshold/d;

    DEBUG_PRINT(printf("compute_converged:\n"));

    checkCudaErrors(cudaMemset(d_converged, 1, sizeof(bool))); //init to true

    compute_converged<<< (k *d / 512)+1, 512 >>>(k, d, l1_thresh, old_centroids, new_centroids, d_converged);

    getLastCudaError("compute_converged() execution failed\n");
    // checkCudaErrors(cudaDeviceSynchronize());
    DEBUG_OUT(" ");

    bool converged;

    cudaMemcpy(&converged, d_converged, sizeof(converged), cudaMemcpyDeviceToHost);
    iterations++;

    done = (iterations > opts->max_iterations) || converged;
  }

  DEBUG_OUT(iterations > opts->max_iterations ? "Max iterations reached!" : "Converged!");

  cudaEventRecord(out_start, 0);

  checkCudaErrors(cudaMemcpy(point_cluster_ids, d_point_centroid_ids, n_points * sizeof(int),
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(*centroids, old_centroids, k * d * sizeof(real),
        cudaMemcpyDeviceToHost));

  cudaEventRecord(out_stop,0);
  cudaEventSynchronize(out_stop);

  float memcpy_d2h_time;
  cudaEventElapsedTime(&memcpy_d2h_time, out_start, out_stop); //TODO: is async messing it up?
  TIMING_PRINT(printf("Device to host: %f ms \n", memcpy_d2h_time));

  DEBUG_OUT("CUDA free");
  checkCudaErrors(cudaFree(d_points));
  checkCudaErrors(cudaFree(old_centroids));
  checkCudaErrors(cudaFree(new_centroids));
  checkCudaErrors(cudaFree(point_centroid_distances));
  checkCudaErrors(cudaFree(d_point_centroid_ids));
  checkCudaErrors(cudaFree(d_k_counts));
  checkCudaErrors(cudaFree(d_converged));

  checkCudaErrors(cudaDeviceSynchronize());

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  float global_time;
  cudaEventElapsedTime(&global_time, start, stop); //TODO: is async messing it up?
  *per_iteration_time = (global_time/iterations);
  TIMING_PRINT(printf("Overall: %f ms \n", global_time));
  TIMING_PRINT(printf("Percent spent in IO: %f \n", (memcpy_d2h_time + memcpy_h2d_time) / global_time));
  TIMING_PRINT(printf("Per iteration: %f ms \n", *per_iteration_time));

  return iterations;
}
