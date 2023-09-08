//
//  kmeans_seq.cpp
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//

#include <iostream>     // std::cout
#include <argparse.h>
#include <io.h>
// #include "argparse.h"
//#include "argparse.hpp"
#include <cmath>        /* pow */
#include <vector>       // std::vector
#include <iterator>
#include <algorithm>    // std::min_element, std::max_element, std::transform
#include <functional>   // std::plus
#include <chrono>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
   next = next * 1103515245 + 12345;
   return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
   next = seed;
}

auto distance_centers(int dimensions,
                      int num_centroids,
                      std::vector<std::vector<float>> old_centers,
                      std::vector<std::vector<float>> new_centers) {

   // Create 2D vector to save euclidean distances of points to each centroids
   std::vector<float> center_distances;

   // Calculate euclidean distance and populate the distance array
   float sum;
   for (int k = 0; k < num_centroids; k++) {
      sum = 0.0;
      for (int d = 0; d < dimensions; d++) {
         sum += pow((old_centers[k][d] - new_centers[k][d]), 2.0);
      }
      center_distances.push_back(sqrt(sum));
   }

   // Get max distance from center_distances to compare to the threshold
   auto max_distance = *std::max_element(center_distances.begin(), center_distances.end());

   return max_distance;
}

std::vector<std::vector<float>> euclidean_distance(int num_points,
                                                   int dimensions,
                                                   int num_centroids,
                                                   std::vector<std::vector<float>> centers,
                                                   const std::vector<std::vector<float>> datapoints) {

   // Create 2D vector to save euclidean distances of points to each centroids
   std::vector<std::vector<float>> distance;

   // Calculate euclidean distance and populate the distance array
   float sum;
   for (int i = 0; i < num_points; i++) {
      distance.push_back(std::vector<float>());
      for (int k = 0; k < num_centroids; k++) {
         sum = 0.0;
         for (int d = 0; d < dimensions; d++) {
            sum += pow((centers[k][d] - datapoints[i][d]), 2.0);
         }
         distance[i].push_back(sqrt(sum));
      }
   }

   return distance;
}

std::vector<int> assign_labels(std::vector<std::vector<float>> distances, int num_points) {
   std::vector<int> labels;
   for (int i = 0; i < num_points; i++) {
      labels.push_back(distance(distances[i].begin(), min_element(distances[i].begin(), distances[i].end())));
   }
   return labels;
}

std::vector<std::vector<float>> recalculate_centers(std::vector<int> labels,
                                                    int num_centroids,
                                                    int num_points,
                                                    int num_dims,
                                                    std::vector<std::vector<float>> centers,
                                                    const std::vector<std::vector<float>> datapoints)
{
   // Make subsets of points
   for (int k=0; k < num_centroids; k++) {
      // Subset of points S_k for each cluster/label
      std::vector<std::vector<float>> S_k;
      int num_points_in_S_k = 0;
      // Iterate through all points and put points into subset S_k according to its label
      for (int j=0; j < num_points; j++) {
         if (labels[j] == k) {
            S_k.push_back(datapoints[j]);
            num_points_in_S_k++;
         }
      }


      if (num_points_in_S_k != 0) {
         // Calculate new averages within each subset of points per cluster
         fill(centers[k].begin(), centers[k].end(), 0.0);
         for (int l=1; l < num_points_in_S_k; l++) {
            std::transform(S_k[0].begin(), S_k[0].end(), S_k[l].begin(), S_k[0].begin(), std::plus<float>());
         }
         // Update centers vector with sums of points in each subset
         centers[k] = S_k[0];

         // Divide the values in centers[] by num_points in subset S_k to get the average values
         float scale = 1.0f / num_points_in_S_k;
         for (auto &value : centers[k]) {
            value = value * scale;
         }
      }
   }
   return centers;
}

auto k_means_sequential(int num_centroids,
                        int num_dims,
                        int max_iterations,
                        float threshold,
                        int cmd_seed,
                        int num_points,
                        const std::vector<std::vector<float>> datapoints)
{
   // Local struct to return as results of k-means
   struct k_means_results {
      std::vector<std::vector<float>> centroids;
      std::vector<int> labels;
      int num_iter_to_converge;
   };

   // Create centers 2D and labels 1D vectors to return as k_means_results
   std::vector<std::vector<float>> centers;
   //   auto centers;
   std::vector<int> labels;

   // Randomly select k data points as centers (initialization)
   kmeans_srand(cmd_seed); // cmd_seed is a cmdline arg
   for (int i=0; i < num_centroids; i++){
      int index = kmeans_rand() % num_points;
      centers.push_back(datapoints[index]);
   }

   int num_iter_to_converge = 0;
   for (int i=0; i < max_iterations; i++) {
      // Measure euclidean distances of points from all centroids
      std::vector<std::vector<float>> distance = euclidean_distance(num_points,
                                                                    num_dims,
                                                                    num_centroids,
                                                                    centers,
                                                                    datapoints);

      // Update clustering labels
      labels = assign_labels(distance, num_points);

      // Update new centers
      std::vector<std::vector<float>> new_centers = recalculate_centers(labels,
                                                                        num_centroids,
                                                                        num_points,
                                                                        num_dims,
                                                                        centers,
                                                                        datapoints);

      // Check convergence
      auto max_distance_of_centers = distance_centers(num_dims, num_centroids,
                                                      centers, new_centers);

      centers = new_centers;
      num_iter_to_converge++;

      if (max_distance_of_centers < threshold) {
         break;
      }
   }

   // Return k-means results
   return k_means_results { centers, labels, num_iter_to_converge };
}

int main(int argc, char **argv) 
{
	// Parse args
	struct options_t opts;
	get_opts(argc, argv, &opts);
	
   // Setup args & read input data
   int n_points;
   std::vector<std::vector<float>> input_vals;
   read_file(&opts, &n_points, input_vals);

   // Start timer
   auto start = std::chrono::high_resolution_clock::now();

   // Get k-means centers, labels, and num_iterations for convergence
   auto [k_means_centers,
         k_means_labels,
         iter_to_converge ] = k_means_sequential(opts.n_cluster, opts.n_dims,
                                                 opts.max_iter, opts.threshold,
                                                 opts.seed, n_points, input_vals);

   //End timer and print out elapsed
   auto end = std::chrono::high_resolution_clock::now();
   auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   std::cout << "time in milliseconds: " << diff.count() << std::endl;
//   std::cout << "iter_to_converge: " << iter_to_converge << std::endl;
//   std::cout << "time_per_iter_in_ms: " << diff.count() / num_iter_conv << std::endl;
   auto time_per_iter_in_ms = diff.count() / iter_to_converge;
//   printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
   printf("%d,%ld\n", iter_to_converge, time_per_iter_in_ms);

   if (opts.centroid) {
      // -c specified, output the centroids of final clusters
      for (int clusterId = 0; clusterId < opts.n_cluster; clusterId ++){
         printf("%d ", clusterId);
         for (int d = 0; d < opts.n_dims; d++)
            printf("%lf ", k_means_centers[clusterId][d]);
         printf("\n");
      }
   } else {
      // -c not specified, output points' label assignment (final cluster id for each point)
      printf("clusters:");
      for (int p=0; p < n_points; p++)
         printf(" %d", k_means_labels[p]);
   }

   return 0;
}
