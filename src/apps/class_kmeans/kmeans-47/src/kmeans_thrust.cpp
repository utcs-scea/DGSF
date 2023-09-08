//
//  kmeans_thrust.cpp
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//

#include <iostream>     // std::cout
// #include <argparse.h>
#include "io.h"
#include "argparse.h"
#include <cmath>        /* pow */
#include <vector>       // std::vector
#include <iterator>
#include <algorithm>    // std::min_element, std::max_element, std::transform
#include <functional>   // std::plus
#include <chrono>

//*****
#include "kmeans_thrust.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
   next = next * 1103515245 + 12345;
   return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
   next = seed;
}

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &original_vector)
{
   std::vector<T> flattened_vector;
   for(const auto &v: original_vector)
      flattened_vector.insert(flattened_vector.end(), v.begin(), v.end());
   return flattened_vector;
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

   // Create centers 2D and labels 1D vectors to return as k_means_results
   std::vector<std::vector<float>> centers;
   std::vector<int> labels(n_points);

   // Randomly select k data points as centers (initialization)
   kmeans_srand(opts.seed); // cmd_seed is a cmdline arg
   for (int i=0; i < opts.n_cluster; i++){
      int index = kmeans_rand() % n_points;
      centers.push_back(input_vals[index]);
   }
   
   // Flatten centers and datapoints vectors
   auto flattened_datapoints = flatten(input_vals);
   auto flattened_centers = flatten(centers);
   
   int iter_to_converge;
   float thrust_time_per_iter_in_ms;
   float thrust_total_time_in_ms;
   std::tie(iter_to_converge, thrust_total_time_in_ms) = k_means_thrust(opts.n_cluster, opts.n_dims, opts.max_iter, opts.threshold, n_points, flattened_centers, flattened_datapoints, labels); 
   
   //End timer and print out elapsed
   auto end = std::chrono::high_resolution_clock::now();
   auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "total time in milliseconds: " << diff.count() << std::endl;
//    std::cout << "thrust total time in milliseconds: " << thrust_total_time_in_ms << std::endl;   
//    std::cout << "iter_to_converge: " << iter_to_converge << std::endl;
//    std::cout << "thrust_time_per_iter_in_ms: " << thrust_total_time_in_ms / iter_to_converge << std::endl;
//    std::cout << "total_time_per_iter_in_ms: " << diff.count() / iter_to_converge << std::endl;
   thrust_time_per_iter_in_ms = thrust_total_time_in_ms / iter_to_converge;
   
   printf("%d,%lf\n", iter_to_converge, thrust_time_per_iter_in_ms);

   if (opts.centroid) {
      // -c specified, output the centroids of final clusters
      for (int clusterId = 0; clusterId < opts.n_cluster; clusterId ++){
         printf("%d ", clusterId);
         for (int d = 0; d < opts.n_dims; d++)
            printf("%lf ", flattened_centers[(opts.n_dims * clusterId) + (d % opts.n_dims)]);
         printf("\n");
      }
   } else {
      // -c not specified, output points' label assignment (final cluster id for each point)
      printf("clusters:");
      for (int p=0; p < n_points; p++)
         printf(" %d", labels[p]);
   }

   return 0;
}
