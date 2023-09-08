//
//  kmeans_thrust.h
//
//  Created by Heejong Jang O'Keefe.
//  Copyright © 2020 Hee Jong Jang O'Keefe. All rights reserved.
//

#ifndef kmeans_thrust_h
#define kmeans_thrust_h

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

std::tuple<int, float> k_means_thrust(int num_centroids,
                        int num_dims,
                        int max_iterations,
                        float threshold,
                        int num_points,
                        std::vector<float>& centers,
                        const std::vector<float>& datapoints,
                              std::vector<int>& labels);

#endif /* kmeans_thrust_h */
