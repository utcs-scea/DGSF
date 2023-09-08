#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "common/common_functions.h"

struct avegage_op : public thrust::unary_function<thrust::tuple<float, int>, float>
{
  __host__ __device__
  float operator()(const thrust::tuple<float, int> d1){
        return thrust::get<0>(d1)/((float) thrust::get<1>(d1));
    }
};

struct centroid_recompute_key_generator : public thrust::unary_function<int, int>
{
  int *centroid_assignments;;
  int dims;
  int num_cluster;

  centroid_recompute_key_generator(int _dims, int* _centroid_assignments) : dims(_dims), centroid_assignments(_centroid_assignments) {};

  __host__ __device__ int operator()(const int i) const {
        return i%dims + dims*(centroid_assignments[i/(dims)]);
    }
};


struct centroid_recompute_reduce_op
{
  __host__ __device__
  thrust::tuple<float, int> operator()(const thrust::tuple<float, int> d1, const thrust::tuple<float, int> d2){
        return thrust::make_tuple(
            thrust::get<0>(d1) + thrust::get<0>(d2),
            thrust::get<1>(d1) + thrust::get<1>(d2)
        );
    }
};

struct centroid_transform : public thrust::unary_function<int, int>
{
  int dims;
  int num_cluster;

  centroid_transform(int _dims, int _num_cluster) : dims(_dims), num_cluster(_num_cluster) {};

  __host__ __device__ int operator()(const int i) const {
    return (i % (dims*num_cluster));
    }
};

struct convergence_op : public thrust::unary_function<float, bool>
{
  float squared_threshold;

  convergence_op(float _squared_threshold) : squared_threshold(_squared_threshold) {};

  __host__ __device__ bool operator()(const float i) const {
    return i<squared_threshold;
    }
};

struct distance_comparator
{
  __host__ __device__
  thrust::tuple<float, int> operator()(const thrust::tuple<float, int> d1, const thrust::tuple<float, int> d2){
    thrust::tuple<float, int> res;
    if (thrust::get<0>(d1) > thrust::get<0>(d2)){
      thrust::get<0>(res) = thrust::get<0>(d2);
      thrust::get<1>(res) = thrust::get<1>(d2);}
    else {
      thrust::get<0>(res) = thrust::get<0>(d1);
      thrust::get<1>(res) = thrust::get<1>(d1);}
    return res;
    }
};

struct division_operator : public thrust::unary_function<int, int>
{
  int divisor;

  division_operator(int _divisor) : divisor(_divisor) {};

  __host__ __device__ int operator()(const int i) const {
    return i/divisor;
    }
};

struct l2_distance : public thrust::unary_function<thrust::tuple<float, float>, float>
{
  __host__ __device__ float operator()(const thrust::tuple<float, float> &tuple) const {
    float diff = thrust::get<0>(tuple) - thrust::get<1>(tuple);
    return diff*diff;
  }
};

struct modulo_transform : public thrust::unary_function<int, int>
{
  int mod;

  modulo_transform(int _mod) : mod(_mod) {};

  __host__ __device__ int operator()(const int i) const {
    return i%mod;
    }
};

struct points_transform : public thrust::unary_function<int, int>
{
  int dims;
  int num_cluster;

  points_transform(int _dims, int _num_cluster) : dims(_dims), num_cluster(_num_cluster) {};

  __host__ __device__ int operator()(const int i) const {
    return i%dims + dims*(i/(num_cluster*dims));
    }
};


int main(int argc, char *argv[]){
    cudaSetDevice(0);
    struct options_t opts;
    int n_vals;
    float *h_points, *h_centroids;
    get_opts(argc, argv, &opts);
    read_file(&opts, &n_vals, &h_points);
    get_initial_centroids(&opts, &n_vals, h_points, &h_centroids);
    
    int point_array_size = n_vals*opts.dims;
    int centroid_array_size = opts.num_cluster*opts.dims;

    thrust::device_vector<float> d_points(h_points, h_points+point_array_size), d_points_temp(n_vals*opts.dims);
    thrust::device_vector<float> d_centroids(h_centroids, h_centroids+centroid_array_size), d_centroids_old(centroid_array_size);

    thrust::device_vector<float> d_points_centroids_distance(n_vals*opts.num_cluster), d_centroid_distances(n_vals), d_convergence_distances(opts.num_cluster);
    thrust::device_vector<int> d_centroid_mappings(n_vals*opts.dims),  d_centroid_assignments(n_vals);

    auto start = std::chrono::high_resolution_clock::now();

    int iters=0;
    for(;iters<opts.max_num_iter;iters++){

        thrust::copy(d_points.begin(), d_points.end(), d_points_temp.begin());
        thrust::copy(d_centroids.begin(), d_centroids.end(), d_centroids_old.begin());

        thrust::reduce_by_key(
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(0),
                division_operator(opts.dims)
            ),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(n_vals*opts.num_cluster*opts.dims),
                division_operator(opts.dims)
            ),
            thrust::make_transform_iterator(
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::make_permutation_iterator(
                            d_centroids.begin(), 
                            thrust::make_transform_iterator(
                                thrust::make_counting_iterator<int>(0), 
                                centroid_transform(opts.dims, opts.num_cluster)
                            )
                        ), 
                        thrust::make_permutation_iterator(
                            d_points.begin(), 
                            thrust::make_transform_iterator(
                                thrust::make_counting_iterator<int>(0), 
                                points_transform(opts.dims, opts.num_cluster)
                            )
                        )
                    )
                ), l2_distance()
            ),
            thrust::make_discard_iterator(),
            d_points_centroids_distance.begin()
        );

        cudaDeviceSynchronize();

        thrust::reduce_by_key(
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(0),
                division_operator(opts.num_cluster)
            ),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(n_vals*opts.num_cluster),
                division_operator(opts.num_cluster)
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_points_centroids_distance.begin(), 
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0), 
                        modulo_transform(opts.num_cluster)
                    )
                )
            ),
            thrust::make_discard_iterator(),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_centroid_distances.begin(), 
                    d_centroid_assignments.begin()
                )
            ),
            thrust::equal_to<int>(),
            distance_comparator()
        );

        cudaDeviceSynchronize();

        thrust::transform(
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_vals*opts.dims),
            d_centroid_mappings.begin(),
            centroid_recompute_key_generator(opts.dims, thrust::raw_pointer_cast(d_centroid_assignments.data()))
        );

        cudaDeviceSynchronize();

        thrust::sort_by_key(
            d_centroid_mappings.begin(),
            d_centroid_mappings.end(),
            d_points_temp.begin()
        );
        
        cudaDeviceSynchronize();

        thrust::reduce_by_key(
            d_centroid_mappings.begin(),
            d_centroid_mappings.end(),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_points_temp.begin(),
                    thrust::make_constant_iterator<int>(1)
                )
            ),
            thrust::make_discard_iterator(),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_centroids.begin(),
                    d_centroid_mappings.begin()
                )
            ),
            thrust::equal_to<int>(),
            centroid_recompute_reduce_op()
        );
        cudaDeviceSynchronize();
        thrust::transform(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_centroids.begin(),
                    d_centroid_mappings.begin()
                )
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_centroids.begin(),
                    d_centroid_mappings.begin()
                )
            ) + centroid_array_size,
            d_centroids.begin(),
            avegage_op()
        );

        cudaDeviceSynchronize();

        thrust::host_vector<float> centroids = d_centroids, old_centroids = d_centroids_old;

        if(test_convergence(&opts, &old_centroids[0], &centroids[0])){
            iters++;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("%d,%lf\n", iters, (double) diff.count() / (double) iters / 1000.0);
    if (opts.output_centroids) {
        thrust::host_vector<float> centroids = d_centroids;
        print_cluster_centroids(&opts, &centroids[0]);
    } else {
        thrust::host_vector<int> centroid_assignments = d_centroid_assignments;
        print_cluster_mappings(&n_vals, &centroid_assignments[0]);
    }

    free(h_points);
    free(h_centroids);

    return 0;    

}