//
//  kmeans_thrust.cu
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//  Using the Nvidia samples with attribution below
//

#include "kmeans_thrust.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <cmath>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <iostream>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <tuple>
#include <vector>

using namespace thrust::placeholders;

// combined structs from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples
// https://github.com/NVIDIA/thrust/blob/main/examples/norm.cu
// https://github.com/NVIDIA/thrust/blob/main/testing/zip_iterator_scan.cu
typedef thrust::tuple<float, float> pt_ct_tuple;
struct distance_squared : public thrust::unary_function<pt_ct_tuple, float>
{
   __host__ __device__ 
   float operator()(const pt_ct_tuple &pt_ctr_tuple) const 
   {
      float distance = thrust::get<0>(pt_ctr_tuple) - thrust::get<1>(pt_ctr_tuple);
      return distance * distance;
   }
};

// compute square root of a number
// from Nvidia thrust doc example in https://thrust.github.io/doc/classthrust_1_1transform__iterator.html
struct square_root : public thrust::unary_function<float,float>
{
   __host__ __device__
   float operator()(float x) const
   {
       return sqrtf(x);
   }
};

// divide sums of points of diff dimensions in centers vector by the number of points in subsets
// adapted from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples
// https://github.com/NVIDIA/thrust/blob/main/testing/zip_iterator_scan.cu
// https://github.com/NVIDIA/thrust/blob/main/examples/simple_moving_average.cu
typedef thrust::tuple<float, int> sums_n_pts_tuple;
struct divide_sums : thrust::unary_function<sums_n_pts_tuple, float>
{
    __host__ __device__
    float operator()(const sums_n_pts_tuple &sums_n_pts_tuple) const
    {
       return (thrust::get<0>(sums_n_pts_tuple) / thrust::get<1>(sums_n_pts_tuple));
    }
};

// convert a linear index to a row or column index
// from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples/sum_rows.cu
template <typename T>
struct linear_index_to_row_col_index : public thrust::unary_function<T,T>
{
  T C_R; // number of columns or rows
  
   __host__ __device__
   linear_index_to_row_col_index(T C_R) : C_R(C_R) {}

   __host__ __device__
   T operator()(T i)
   {
      return i / C_R;
   }
};

template <typename T>
struct index_seq_pts : public thrust::unary_function<T,T>
{
   T C;
   T R;
   T* vec_ptr;
   
   index_seq_pts(T C, T R, T* vec_ptr) : C(C), R(R), vec_ptr(vec_ptr) {}

   __host__ __device__ 
   T operator()(const T i) const 
   {
      return (i / R) + C * vec_ptr[i % R];
   }
};

// convert linear index into column-major order index sequence
// adapted from linear_index_to_row_index in Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples/sum_rows.cu
template <typename T>
struct index_sequence_points : public thrust::unary_function<T,T>
{
   T C;
   T R;

   index_sequence_points(T C, T R) : C(C), R(R) {}

   __host__ __device__ 
   T operator()(const T i) const 
   {
      return (i % R) * C + (i / R);
   }
};

// convert linear index into row-major index sequence
// adapted from Nvidia thrust doc examples 
// https://github.com/NVIDIA/thrust/blob/main/examples/sum_rows.cu
// https://github.com/NVIDIA/thrust/blob/master/examples/expand.cu
template <typename T>
struct index_sequence_centers : public thrust::unary_function<T,T>
{
   T C;
   T R;

   index_sequence_centers(T C, T R) : C(C), R(R) {}

   __host__ __device__ 
   T operator()(const T i) const 
   {
       return (i % C) + ( C * (i / (R * C)));
    }
};

auto distance_centers_thrust(int num_dims,
                             int num_centroids,
                             const thrust::device_vector<float>& old_centers,
                             const thrust::device_vector<float>& new_centers) 
{
   // Create a vector to save euclidean distances of points to each centroids
   thrust::device_vector<float> center_distances(num_centroids);
   
   // Calculate euclidean distance and populate the distance array
   auto key_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_dims));
   auto key_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_dims)) + (num_centroids * num_dims);
   auto old_new_centers_tuple = thrust::make_zip_iterator(thrust::make_tuple(old_centers.begin(), new_centers.begin()));
   auto center_distances_squared = thrust::make_transform_iterator(old_new_centers_tuple, distance_squared());
   thrust::reduce_by_key(key_first, key_last, center_distances_squared, thrust::make_discard_iterator(), center_distances.begin());
   thrust::transform(center_distances.begin(), center_distances.end(), center_distances.begin(), square_root());
   
   // Get max distance from center_distances to compare to the threshold
   auto max_distance = thrust::max_element(center_distances.begin(), center_distances.end());
   float max_distance_host = *max_distance;
   
   return max_distance_host;
}

auto euclidean_distance_thrust(int num_points,
                               int num_dims,
                               int num_centroids,
                               const thrust::device_vector<float>& centers,
                               const thrust::device_vector<float>& datapoints,
                               thrust::device_vector<float>& distances) 
{
   // num_dims = number of columns = C
   // num_points = number of rows = R
   
   // Calculate euclidean distance and populate the distances vector
   // adapted from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples/sum_rows.cu
   auto key_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_dims));
   auto key_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_dims)) + (num_dims * num_points * num_centroids);
   auto centroid_index_sequence = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_sequence_centers<int>((num_centroids * num_dims), num_points));
   auto point_index_sequence = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_sequence_centers<int>(num_dims, num_centroids));   
   auto centroid_of_tuple = thrust::make_permutation_iterator(centers.begin(), centroid_index_sequence);
   auto pointer_of_tuple = thrust::make_permutation_iterator(datapoints.begin(), point_index_sequence);
   auto pointer_centroid_tuple = thrust::make_zip_iterator(thrust::make_tuple(centroid_of_tuple, pointer_of_tuple));
   auto distances_squared = thrust::make_transform_iterator(pointer_centroid_tuple, distance_squared());
   
   thrust::reduce_by_key(key_first, key_last, distances_squared, thrust::make_discard_iterator(), distances.begin());
   thrust::transform(distances.begin(), distances.end(), distances.begin(), square_root());
}

auto assign_labels_thrust(thrust::device_vector<float>& distances, 
                          int num_points,
                          int num_centroids,
                          thrust::device_vector<int>& labels) 
{
   // num_centroids = number of columns = C
   // num_points = number of rows = R
   
   // Calculate euclidean distance and populate the distances vector
   // referred and adapted from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples/sum_rows.cu
   // referred and adapted https://stackoverflow.com/questions/23970593/get-nearest-centroid-using-thrust-library-k-means
   auto key_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_centroids));
   auto key_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_centroids)) + (num_points * num_centroids);
   auto distance_index_sequence = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_sequence_centers<int>(num_centroids, num_points));
   auto distances_tuple = thrust::make_zip_iterator(thrust::make_tuple(distances.begin(), distance_index_sequence));
   auto labels_tuple = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), labels.begin()));
   thrust::reduce_by_key(key_first, key_last, distances_tuple, thrust::make_discard_iterator(), labels_tuple, thrust::equal_to<int>(), thrust::minimum<thrust::tuple<float, int>>());
}

auto recalculate_centers_thrust(const thrust::device_vector<int>& labels,
                                int num_centroids,
                                int num_points,
                                int num_dims,
                                thrust::device_vector<float>& centers,
                                const thrust::device_vector<float>& datapoints)
{  
   // Do a key-value sort according to the ascending order of cluster labels ID and its corresponding point
   thrust::counting_iterator<int> iter(0);
   thrust::device_vector<int> points_index_seq(num_points);
   thrust::copy(iter, iter + points_index_seq.size(), points_index_seq.begin());
   thrust::device_vector<int> labels_copy = labels;
   thrust::sort_by_key(labels_copy.begin(), labels_copy.end(), points_index_seq.begin());
   
   // Create a vector to populate the points in each subset of points S_k for each cluster/label
   thrust::device_vector<int> num_points_in_subsets(num_centroids);
   thrust::reduce_by_key(labels_copy.begin(), labels_copy.end(), thrust::constant_iterator<int>(1), thrust::make_discard_iterator(), num_points_in_subsets.begin());

   // Calculate new averages within each subset of points per cluster by putting sums of subsets S_k of each centroid
   // Update centers vector with sums of points in each subset
   // Scatter-gather with column-major order, gather all the dimensions/columns together
   auto key_first_param_2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_sequence_centers<int>(num_points, num_dims));   
   auto key_first = thrust::make_permutation_iterator(labels_copy.begin(), key_first_param_2);
   auto key_last = key_first + (num_points * num_dims);
   auto values_in_param_2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_seq_pts<int>(num_dims, num_points, thrust::raw_pointer_cast(&points_index_seq[0])));
   auto values_in = thrust::make_permutation_iterator(datapoints.begin(), values_in_param_2);
   auto values_out_param_2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), index_sequence_points<int>(num_dims, num_centroids));
   auto values_out = thrust::make_permutation_iterator(centers.begin(), values_out_param_2);
   thrust::reduce_by_key(key_first, key_last, values_in, thrust::make_discard_iterator(), values_out);
   // centers now updated with sums of subsets S_k of each centroid
   
   // Divide the values in centers vector by num_points in subset S_k to get the average values
   // adapted from Nvidia thrust doc examples https://github.com/NVIDIA/thrust/blob/main/examples/simple_moving_average.cu#L54
   auto indices = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_col_index<int>(num_dims));
   auto n_pts_subset_idx_seq = thrust::make_permutation_iterator(num_points_in_subsets.begin(), indices);
   auto input_seq = thrust::make_zip_iterator(thrust::make_tuple(centers.begin(), n_pts_subset_idx_seq));
   thrust::transform(input_seq, input_seq + (num_centroids * num_dims), centers.begin(), divide_sums());   
}

std::tuple<int, float> k_means_thrust(int num_centroids,
                    int num_dims,
                    int max_iterations,
                    float threshold,
                    int num_points,
                    std::vector<float>& centers,
                    const std::vector<float>& datapoints,
                   std::vector<int>& labels)
{      
   // Create device vectors and copy host_vector_data into them
   thrust::device_vector<float> centers_device = centers;
   const thrust::device_vector<float> datapoints_device = datapoints;
   
   // Device vectors to populate distances, labels, new_centers
   thrust::device_vector<float> distances_device(num_points * num_centroids);
   thrust::device_vector<int> labels_device(num_points);
   thrust::device_vector<float> new_centers_device = centers_device;
   
   int num_iter_to_converge = 0;
   
   // Start timer
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   
   for (int i=0; i < max_iterations; i++) {
            
      // Measure euclidean distances of points from all centroids
      euclidean_distance_thrust(num_points, num_dims, num_centroids, centers_device, datapoints_device, distances_device);

      // Update clustering labels
      assign_labels_thrust(distances_device, num_points, num_centroids, labels_device);
      
      // Update new centers
      recalculate_centers_thrust(labels_device, num_centroids, num_points, num_dims, new_centers_device, datapoints_device);

      // Check convergence
      auto max_distance_of_centers = distance_centers_thrust(num_dims, num_centroids, centers_device, new_centers_device);
      
      centers_device = new_centers_device;
      num_iter_to_converge++;

      if (max_distance_of_centers < threshold) {
         break;
      }
   }
   
   // End timer and print out elapsed
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);

   // Put labels and centers back to host vectors
   thrust::copy(labels_device.begin(), labels_device.end(), labels.begin());
   thrust::copy(centers_device.begin(), centers_device.end(), centers.begin());  
   
   return std::make_tuple(num_iter_to_converge, milliseconds);
}
