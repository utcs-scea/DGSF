#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <limits>
#include <thrust/device_vector.h>
#include "rng.h"
#include "point.h"

class thrust_kmeans {
    public:
    thrust_kmeans(
        int k,
        int dims,
        std::vector<point>&& dataset,
        std::unique_ptr<rng> randomizer,
        int max_iterations,
        double threshold
    );
    void find();
    void print_centroids();
    void print_labels();
    
    private:
    thrust::device_vector<double> generate_new_centroids_(const thrust::device_vector<int>& labels);
    int nearest_centroid_(const point&);
    bool is_converged_(const thrust::device_vector<double>& previous_centroids);
    
    int n_;
    int k_;
    int dims_;
    std::vector<point> centroids_;
    std::unique_ptr<rng> randomizer_;
    int max_iterations_;
    double convergence_threshold_;
    int iteration_;
    thrust::device_vector<double> dcentroids_;
    thrust::device_vector<double> ddataset_;
    thrust::device_vector<int> dcentroid_idxs_;
    thrust::device_vector<int> ddataset_idxs_;
};