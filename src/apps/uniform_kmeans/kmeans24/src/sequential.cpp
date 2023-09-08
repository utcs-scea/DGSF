#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "common.h"


double get_euclidean_distance_squared(
    const std::vector<float> &points,
    const std::vector<float> &centroids,
    int point_idx,
    int centroid_idx,
    int dims)
{
    double dist = 0.0;

    for (int j = 0; j < dims; ++j) {
        float diff =
            points[point_idx * dims + j] - centroids[centroid_idx * dims + j];

        dist += std::pow(diff, 2);
    }

    return dist;
}

void find_nearest_centroids(
    const std::vector<float> &points,
    const std::vector<float> &centroids,
    std::vector<int> &labels,
    const options_t &opts)
{
    for (unsigned int i = 0; i < (points.size() / opts.dims); ++i) {
        float min_dist = std::numeric_limits<float>::max();
        int min_centroid = -1;
        for (int j = 0; j < opts.num_cluster; ++j) {
            float dist = get_euclidean_distance_squared(
                points, centroids, i, j, opts.dims);

            if (dist < min_dist) {
                min_dist = dist;
                min_centroid = j;
            }
        }
        labels[i] = min_centroid;
    }
}

void update_centroids(
    const std::vector<float> &points,
    std::vector<float> &centroids,
    const std::vector<int> &labels,
    const options_t &opts)
{
    for (auto &centroid : centroids) {
        centroid = 0.0;
    }

    std::vector<int> centroid_label_count(opts.num_cluster, 0);

    for (unsigned int i = 0; i < labels.size(); ++i) {
        int centroid_id = labels[i];
        for (int dim = 0; dim < opts.dims; ++dim) {
            centroids[centroid_id * opts.dims + dim] += points[i * opts.dims + dim];
        }
        centroid_label_count[centroid_id] += 1;
    }

    for (int i = 0; i < opts.num_cluster; ++i) {
        if (centroid_label_count[i] > 0) {
            for (int dim = 0; dim < opts.dims; ++dim) {
                centroids[i * opts.dims + dim] /= centroid_label_count[i];
            }
        }
    }
}

bool converged(
    const std::vector<float> &old_centroids,
    const std::vector<float> &centroids,
    const options_t &opts)
{
    for (int i = 0; i < opts.num_cluster; ++i) {
        int dist = get_euclidean_distance_squared(
            centroids, old_centroids, i, i, opts.dims);

        if (std::sqrt(dist) > opts.threshold) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    struct options_t opts;
    get_opts(argc, argv, &opts);

    std::vector<float> points;
    int n_points;
    read_file(opts, n_points, points);

    std::vector<float> centroids(opts.num_cluster * opts.dims);
    generate_centroids(points, centroids, opts);
    std::vector<int> labels(n_points);
    std::vector<float> old_centroids(centroids.size());

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    int i = 0;
    while (++i < opts.max_num_iter) {
        old_centroids = centroids;
        find_nearest_centroids(points, centroids, labels, opts);
        update_centroids(points, centroids, labels, opts);
        bool done = converged(old_centroids, centroids, opts);

        if (done) {
            break;
        }
    }

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time_in_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    output_results(
        labels,
        centroids,
        static_cast<float>(total_time_in_ms.count()),
        i,
        opts);

    return 0;
}