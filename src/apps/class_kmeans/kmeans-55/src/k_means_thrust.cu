#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/logical.h>

#include "k_means_thrust.h"
#include "common.h"

typedef thrust::device_vector<real> dv_real;
typedef thrust::device_vector<int> dv_int;
typedef thrust::host_vector<real> hv_real;

// referenced from piazza, which references NVidia
__device__ double datomicAdd(double* address, double val)
{
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

#define D_PRINT_POINT(x, d, i) { \
  if(DEBUG_TEST) {\
    thrust::copy_n( \
      x.begin() + i*d, \
      d, \
      std::ostream_iterator<real>(std::cerr, ", ") \
    ); \
    std::cerr << std::endl;\
  }\
}
#define D_PRINT_ALL(x) { \
  if(DEBUG_TEST) {\
    thrust::copy( \
      x.begin(), \
      x.end(), \
      std::ostream_iterator<real>(std::cerr, ", ") \
    ); \
    std::cerr << std::endl;\
  }\
}

typedef thrust::tuple<real, int> real_indexed;
typedef thrust::tuple<int, int> two_ints;
typedef thrust::tuple<real, real> two_realz;

// map an index -> component l of point index i and centroid index j
// and then compute the distance component of l
struct distance_component : public thrust::unary_function<int, real> {
  const int d, k;
  const real *points, *centroids;

  distance_component(int _d, int _k, real *_points, real *_centroids)
    : d(_d), k(_k), points(_points), centroids(_centroids) {}

  __host__ __device__
  real operator()(int index) {
    int i = (index / d) / k;
    int j = (index / d) % k;
    int l = index % d;

    // TODO:
    return POW2(points[i*d+l] - centroids[j*d+l]);
    // return POW2(points[i*d+l]) - 2*points[i*d+l]*centroids[j*d+l] + POW2(centroids[j*d+l]);
  }
};

// sum up each point component into a new centroid
struct centroid_means : public thrust::unary_function<void, real_indexed> {
  const int d;
  const int *d_point_cluster_ids;
  real *centroids;

  centroid_means(int _d, int *_d_point_cluster_ids, real *_centroids)
    : d(_d), d_point_cluster_ids(_d_point_cluster_ids),
    centroids(_centroids) {}

  __device__
  void operator()(real_indexed real_index) {
    real value = thrust::get<0>(real_index);
    int index = thrust::get<1>(real_index);
    int target_centroid_id = d_point_cluster_ids[index / d];
    int target_centroid_component_id = target_centroid_id*d + index % d;
    // printf("%lf %i %i %i\n", value, index, target_centroid_id, target_centroid_component_id);

#ifdef REAL_FLOAT
    atomicAdd(centroids + target_centroid_component_id, value);
#else
    datomicAdd(centroids + target_centroid_component_id, value);
#endif
  }
};

struct centroid_divide : public thrust::unary_function<void, two_ints> {
  const int d, i;
  real *centroids;

  centroid_divide(int _d, int _i, real *_centroids)
    : d(_d), i(_i), centroids(_centroids) {}

  __device__
  void operator()(two_ints ints) {
    int centroid_id = thrust::get<0>(ints);
    int counts = thrust::get<1>(ints);

    *(centroids + centroid_id*d + i) /= counts;
  }
};

//functor that makes point component indexes equivalent based on a map
//in a n_points * k array
struct point_component_centroid_id : public thrust::binary_function<int, int, int> {
  const int d;
  const int *point_cluster_ids;
  const real *centroids;

  point_component_centroid_id(int _d, int *_point_cluster_ids, real *_centroids)
    : d(_d), point_cluster_ids(_point_cluster_ids), centroids(_centroids) {}

  __host__ __device__
  int operator()(int index) {
    return point_cluster_ids[index / d] * d + index % d;
  }
};

//binary function for a maximum on the first member of a tuple
struct maximum_by_first : public thrust::binary_function<real_indexed, real_indexed, real_indexed> {
  maximum_by_first() {}

  __host__ __device__
  real_indexed operator()(real_indexed x_1, real_indexed x_2) {
    return thrust::get<0>(x_1) < thrust::get<0>(x_2) ? x_1 : x_2;
  }
};

struct l1_op : public thrust::unary_function<bool, two_realz> {
  const real l1_thresh;
  l1_op(real _l1_thresh) : l1_thresh(_l1_thresh) {}

  __host__ __device__
  bool operator()(two_realz realz) {
    // (_1 - _2 < 0 ? _2 - _1 : _1 - _2) < l1_thresh
    real x1 = thrust::get<0>(realz);
    real x2 = thrust::get<1>(realz);

    return abs(x1 - x2) < l1_thresh;
  }
};

int k_means_thrust(int n_points, real *points, struct options_t *opts,
  int* point_cluster_ids, real** centroids, double *per_iteration_time) {

  using namespace thrust;

  //https://github.com/NVIDIA/thrust/blob/1d067bdba7aaca4b53cd4d43b98ac180a0308446/examples/lambda.cu
  using namespace thrust::placeholders;

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

  // timer code - 0_Simple/simpleMultiCopy/simpleMultiCopy.cu
  cudaEventRecord(start, 0);
  cudaEventRecord(in_start, 0);

  dv_real d_points(points, points + n_points * d);

  dv_real old_centroids(*centroids, *centroids + k * d);

  cudaEventRecord(in_stop,0);
  cudaEventSynchronize(in_stop);

  float memcpy_h2d_time;
  cudaEventElapsedTime(&memcpy_h2d_time, in_start, in_stop);
  TIMING_PRINT(printf("Host to device: %f ms \n", memcpy_h2d_time));

  dv_real new_centroids(k * d);

  dv_real point_centroid_distances(n_points * k);
  dv_int d_point_cluster_ids(n_points);
  dv_int unsorted_d_point_cluster_ids(n_points);

  dv_int d_k_counts(k);
  dv_int d_k_count_keys(k);

  while(!done) {
    DEBUG_OUT("old centroid");
    D_PRINT_ALL(old_centroids);

    distance_component distance_component(d, k,
        raw_pointer_cast(d_points.data()), raw_pointer_cast(old_centroids.data()));

    // compute the distances for each point/centroid
    // https://github.com/NVIDIA/thrust/blob/1d067bdba7aaca4b53cd4d43b98ac180a0308446/examples/sum_rows.cu
    reduce_by_key(
      make_transform_iterator(counting_iterator<int>(0), _1 / d),
      make_transform_iterator(counting_iterator<int>(n_points * k * d), _1 / d),
      make_transform_iterator(counting_iterator<int>(0),  distance_component),
      make_discard_iterator(), //discard keys
      point_centroid_distances.begin()
    );

    DEBUG_OUT("centroid distance");
    D_PRINT_ALL(point_centroid_distances);

    // reduce to the index of the nearest centroid
    reduce_by_key(
      make_transform_iterator(counting_iterator<int>(0), _1 / k),
      make_transform_iterator(counting_iterator<int>(n_points * k), _1 / k),
      make_zip_iterator(
        make_tuple(
          point_centroid_distances.begin(),
          counting_iterator<int>(0))),
      make_discard_iterator(),
      make_zip_iterator(
        make_tuple(
          make_discard_iterator(), //discard distance values
          unsorted_d_point_cluster_ids.begin())),
      equal_to<int>(),
      maximum_by_first()
    );

    transform(
      unsorted_d_point_cluster_ids.begin(),
      unsorted_d_point_cluster_ids.end(),
      unsorted_d_point_cluster_ids.begin(),
      _1 % k
    );

    DEBUG_OUT("%k unsorted_d_point_cluster_ids");
    D_PRINT_ALL(unsorted_d_point_cluster_ids);

    // zero the new centroids
    fill(new_centroids.begin(), new_centroids.end(), 0);

    // point_component_centroid_id point_component_centroid_id(d,
    //     raw_pointer_cast(d_point_cluster_ids.data()),
    //     raw_pointer_cast(new_centroids.data()));
    // dv_int v(k*d);

    //compute the sum of points for each centroid
    // reduce_by_key(
    //   make_transform_iterator(counting_iterator<int>(0), point_component_centroid_id),
    //   make_transform_iterator(counting_iterator<int>(n_points * d), point_component_centroid_id),
    //   d_points.begin(),
    //   v.begin(),
    //   // make_discard_iterator(), //discard keys
    //   new_centroids.begin()
    // );
    // ok, no fancy reduce_by_key

    centroid_means centroid_means(d,
      raw_pointer_cast(unsorted_d_point_cluster_ids.data()),
      raw_pointer_cast(new_centroids.data()));

    for_each(
      make_zip_iterator(
        make_tuple(d_points.begin(), counting_iterator<int>(0))),
      make_zip_iterator(
        make_tuple(d_points.end(), counting_iterator<int>(n_points * d))),
      centroid_means
    );

    DEBUG_OUT("new_centroids sums");
    D_PRINT_ALL(new_centroids);
    // D_PRINT_ALL(v);

    d_point_cluster_ids = unsorted_d_point_cluster_ids;

    sort(
      d_point_cluster_ids.begin(),
      d_point_cluster_ids.end()
    );

    //compute the point count for each centroid
    auto new_end = reduce_by_key(
      d_point_cluster_ids.begin(),
      d_point_cluster_ids.end(),
      make_constant_iterator(1),
      d_k_count_keys.begin(),
      d_k_counts.begin()
    );

    // zero out any stragglers
    fill(new_end.second, d_k_counts.end(), 0);

    DEBUG_OUT("k_counts");
    D_PRINT_ALL(d_k_counts);
    D_PRINT_ALL(d_k_count_keys);
    // assert(new_end.first == d_k_count_keys.end());

    for (int i = 0; i < d; i++) {
      for_each(
        make_zip_iterator(
          make_tuple(d_k_count_keys.begin(), d_k_counts.begin())),
        make_zip_iterator(
          make_tuple(new_end.first, new_end.second)),
        centroid_divide(d, i, raw_pointer_cast(new_centroids.data()))
      );
    }

    DEBUG_OUT("new_centroids means");
    D_PRINT_ALL(new_centroids);

    // swap centroids
    swap(new_centroids, old_centroids);

    real l1_thresh = opts->threshold/d;

    bool converged = transform_reduce(
      make_zip_iterator(
        make_tuple(new_centroids.begin(), old_centroids.begin())),
      make_zip_iterator(
        make_tuple(new_centroids.end(), old_centroids.end())),
      l1_op(l1_thresh),
      true,
      logical_and<bool>()
    );

    iterations++;
    done = (iterations > opts->max_iterations) || converged;
  }
  // release the other centroids buffer

  DEBUG_OUT(iterations > opts->max_iterations ? "Max iterations reached!" : "Converged!" );

  cudaEventRecord(out_start, 0);

  copy(old_centroids.begin(), old_centroids.end(), *centroids);
  copy(unsorted_d_point_cluster_ids.begin(), unsorted_d_point_cluster_ids.end(), point_cluster_ids);

  cudaEventRecord(out_stop,0);
  cudaEventSynchronize(out_stop);

  float memcpy_d2h_time;
  cudaEventElapsedTime(&memcpy_d2h_time, out_start, out_stop); //TODO: is async messing it up?
  TIMING_PRINT(printf("Device to host: %f ms \n", memcpy_d2h_time));

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
