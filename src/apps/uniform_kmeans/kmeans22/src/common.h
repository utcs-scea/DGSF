#ifndef COMMON_H
#define COMMON_H

#include <getopt.h>
#include <vector>


struct options_t {
    int num_cluster;
    int dims;
    char *in_file;
    int max_num_iter;
    float threshold;
    bool output_centroid;
    int seed;
};

int kmeans_rand();

void kmeans_srand(unsigned int seed);

void get_opts(int argc, char **argv, struct options_t *opts);

void read_file(const struct options_t &opts,
               int &n_points,
               std::vector<float> &points);

void generate_centroids(
    const std::vector<float> &points,
    std::vector<float> &centroids,
    const struct options_t &opts);

void output_results(
    const std::vector<int> &labels,
    const std::vector<float> &centroids,
    float total_time_in_ms,
    int iter_to_converge,
    const struct options_t &opts);

#endif