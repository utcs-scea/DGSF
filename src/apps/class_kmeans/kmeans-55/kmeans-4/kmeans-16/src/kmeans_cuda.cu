//
//  kmeans_cuda.cu
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//


#include "kmeans_cuda.h"
#include <iostream>
#include <tuple>
#include <cfloat>
#include <cmath>        /* sqrt */
#include <vector>       // std::vector
#include <iterator>
#include <algorithm>    
#include <functional>   
#include <chrono>
#include <cuda_runtime.h> 
#include <stdio.h> 
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

__global__ void distance_centers_cuda(int num_dims,
                                      int num_centroids,
                                      const float *old_centers,
                                      const float *new_centers,
                                      float *max_distance) 
{
   // Calculate euclidean distance of centers
   float sum;
   float distance;
   float max_distance_in_device_function = 0.0;
   
   for (int k = 0; k < num_centroids; k++) {
      sum = 0.0;
      distance = 0.0;
      for (int d = 0; d < num_dims; d++) {
         distance = old_centers[k * num_dims + d] - new_centers[k * num_dims + d];
         sum += distance * distance;
      }
      distance = sqrt(sum);
      if (distance > max_distance_in_device_function) {
         max_distance_in_device_function = distance;
      }
   }
   
   *max_distance = max_distance_in_device_function;
}

__device__ float euclidean_distance_cuda(int num_dims,
                                         const float *centers,
                                         const float *datapoints,
                                         int point_id,
                                         int centroid_id) 
{
   // Calculate euclidean distance
   float sum = 0.0;
   float distance = 0.0;
   
   for (int d=0; d < num_dims; d++) {
      distance = centers[centroid_id * num_dims + d] - datapoints[point_id * num_dims + d];
      sum += distance * distance;
   }
   
   return sqrt(sum);
}

__global__ void assign_labels_cuda(int num_points,
                                   int num_dims,
                                   int num_centroids,
                                   const float *centers,
                                   const float *datapoints,
                                   int *labels,
                                   float *sums_of_subsets,
                                   int *num_points_in_subsets) 
{
   // One point per thread
   const int point_id = threadIdx.x + blockIdx.x * blockDim.x;
   
   // Only do something if the point_id is within the range of num_points
   if (point_id < num_points) {
      float min_distance = INFINITY;
      int label = 0;
      
      for (int k = 0; k < num_centroids; k++) {
         float distance = euclidean_distance_cuda(num_dims, centers, datapoints, point_id, k);
         
         if (distance < min_distance) {
            min_distance = distance;
            label = k;
         }
      }
      
      // Assign label to the point that this thread is working on
      labels[point_id] = label;
      
      // Make subsets S_k of points here instead of in the recalculate_centers_cuda since one thread goes over one point
      // and it's easier to add up the sums of point dimension values when each point is assigned its centroid label
      for (int d=0; d < num_dims; d++) {
         atomicAdd(&sums_of_subsets[label * num_dims + d], datapoints[point_id * num_dims + d]);
      }
      // Add up number of points in each subset S_k
      atomicAdd(&num_points_in_subsets[label], 1);
   }
}

__global__ void recalculate_centers_cuda(int num_centroids,
                                         int num_dims,
                                float *centers,
                                const float *sums_of_subsets,
                                        const int *num_points_in_subsets)
{  
   // One centroid per thread
   // Only needed 1 block with threads eqaul to number of clusters/centroids, so centroid_id is the threadId for 1D block
   const int centroid_id = threadIdx.x;
   
   if (centroid_id < num_centroids) {
      for (int d=0; d < num_dims; d++) {
         centers[centroid_id * num_dims + d] = sums_of_subsets[centroid_id * num_dims + d] / num_points_in_subsets[centroid_id];
      }
   }
}

std::tuple<int, float> k_means_cuda(int num_centroids,
                                       int num_dims,
                                       int max_iterations,
                                       float threshold,
                                       int num_points,
                                       std::vector<float>& centers,
                                       const std::vector<float>& datapoints,
                                       std::vector<int>& labels)
{      
   // Create device pointers for vectors, alloc space for device copies, and copy host_vector_data into them
   float *centers_device;
   float *datapoints_device;
   const size_t centers_size = num_centroids * num_dims * sizeof(float); // centers.sizeof()
   const size_t datapoints_size = num_points * num_dims * sizeof(float); // datapoints.sizeof()
   cudaMalloc((float**)&centers_device, centers_size);
   cudaMalloc((float**)&datapoints_device, datapoints_size);
   cudaMemcpy(centers_device, centers.data(), centers_size, cudaMemcpyHostToDevice); // centers.data() = &centers[0]
   cudaMemcpy(datapoints_device, datapoints.data(), datapoints_size, cudaMemcpyHostToDevice); // datapoints.data() = &datapoints[0]
      
   
   // Device pointers for vectors to populate data needed to calculate k-means
   int *labels_device; // (num_points)
   float *new_centers_device; // (num_centroids * num_dims)
   float *sums_of_subsets_device; // sums_of_subsets for each cluster (num_centroids * num_dims)
   int *num_points_in_subsets_device; // (num_centroids)
   float *max_distance_device; // ptr with 1-element
   float *max_distance_host;
   
   // Alloc space for device copies
   const size_t labels_size = num_points * sizeof(int);
   cudaMalloc((int**)&labels_device, labels_size);
   cudaMalloc((float**)&new_centers_device, centers_size);
   cudaMemcpy(new_centers_device, centers.data(), centers_size, cudaMemcpyHostToDevice);
   cudaMalloc((float**)&sums_of_subsets_device, centers_size);
   const size_t num_points_in_subsets_size = num_centroids * sizeof(int);
   cudaMalloc((int**)&num_points_in_subsets_device, num_points_in_subsets_size);
   const size_t max_distance_size = 1 * sizeof(float);
   cudaMalloc((float**)&max_distance_device, max_distance_size);
   max_distance_host = (float *)malloc(max_distance_size);
   
   // Invoke kernel at host side
   const int threads_per_block(num_dims);
   const int blocks_per_grid((num_points + threads_per_block - 1) / threads_per_block);
   
   int num_iter_to_converge = 0;
   
   // Start timer
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   
   for (int i=0; i < max_iterations; i++) {
      
      // Zero-out vectors
      cudaMemset(sums_of_subsets_device, 0, centers_size);
      cudaMemset(num_points_in_subsets_device, 0, num_points_in_subsets_size);
      
      // Update clustering labels
      assign_labels_cuda<<<blocks_per_grid, threads_per_block>>>(num_points, num_dims, num_centroids, centers_device, datapoints_device, labels_device, sums_of_subsets_device, num_points_in_subsets_device);
      cudaDeviceSynchronize();
      
      // Update new centers
      // Only need 1 block with number of threads equal to the number of clusters/centroids, 1 centroid/thread
      recalculate_centers_cuda<<<1, num_centroids>>>(num_centroids, num_dims, new_centers_device, sums_of_subsets_device, num_points_in_subsets_device);
      cudaDeviceSynchronize();

      // Check convergence
      distance_centers_cuda<<<1, 1>>>(num_dims, num_centroids, centers_device, new_centers_device, max_distance_device);
      cudaDeviceSynchronize();
      // Copy max distance from device to host to compare to the threshold
      cudaMemcpy(max_distance_host, max_distance_device, max_distance_size, cudaMemcpyDeviceToHost);
      
      cudaMemcpy(centers_device, new_centers_device, centers_size, cudaMemcpyDeviceToDevice);
      num_iter_to_converge++;

      if (*max_distance_host < threshold) {
         break;
      }
   }
   
   // End timer
   cudaEventRecord(stop);

   // Put labels and centers back to host vectors                 
   cudaMemcpy(labels.data(), labels_device, labels_size, cudaMemcpyDeviceToHost);
   cudaMemcpy(centers.data(), centers_device, centers_size, cudaMemcpyDeviceToHost);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   
   // Free device memory
   cudaFree(centers_device);
   cudaFree(datapoints_device);
   cudaFree(labels_device);
   cudaFree(new_centers_device);
   cudaFree(sums_of_subsets_device);
   cudaFree(num_points_in_subsets_device);
   cudaFree(max_distance_device);
     
   return std::make_tuple(num_iter_to_converge, milliseconds);
}
