//
//  kmeans_cuda.h
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//

#ifndef kmeans_cuda_h
#define kmeans_cuda_h

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <stdio.h> 

std::tuple<int, float> k_means_cuda(int num_centroids,
                        int num_dims,
                        int max_iterations,
                        float threshold,
                        int num_points,
                        std::vector<float>& centers,
                        const std::vector<float>& datapoints,
                              std::vector<int>& labels);

#endif /* kmeans_cuda_h */
