#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <chrono>
#include "rng.h"
#include "point.h"

class kmeans {
    public:
    kmeans(
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
    void print_time();
    
    private:
    std::vector<point> generate_new_centroids_(const std::vector<int>& labels);
    int nearest_centroid_(const point&);
    double distance_(const point& a, const point& b);
    bool is_converged_(const std::vector<point>& previous_centroids);
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    int k_;
    int dims_;
    std::vector<point> dataset_;
    std::vector<point> centroids_;
    std::unique_ptr<rng> randomizer_;
    int max_iterations_;
    double convergence_threshold_;
    int iteration_;
};