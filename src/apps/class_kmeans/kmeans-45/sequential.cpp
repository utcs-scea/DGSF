#include <random>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <iterator>
#include "lib/argparse.h"
#include "lib/datasets.h"
#include "lib/timer.h"

template<typename T>
std::ostream & operator<<(std::ostream & os, std::vector<T> vec)
{
    os<<"{ ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
    os<<"}";
    return os;
}

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

struct kmeans_t {
    std::vector<std::vector<double>> centroids;
    std::vector<int> labels;
};

double dist3(const std::vector<double>& a, const std::vector<double>& b) {
    double acc = 0.0;
    for (int i = 0; i < a.size(); i++) {
        acc += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return acc;
}

int find_nearest_centroid(const std::vector<double>& p, const std::vector<std::vector<double>>& centroids) {
    int label = -1;
    double min_dist = std::numeric_limits<double>::infinity();
    int i = 0;
    for (auto it = centroids.begin(); it != centroids.end(); ++it, ++i) {
        double d = dist3(p, *it);
        if (d < min_dist) {
            min_dist = d;
            label = i;
        }
    }
    return label;
}

bool converged(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>&  b, double threshold) {
    for (int i = 0; i < a.size(); ++i) {
        double d = dist3(a[i], b[i]);
        if (d > threshold * threshold) {
            return false;
        }
    }
    return true;
}

kmeans_t* kmeans(dataset_t* data, args_t* args) {
    kmeans_t* res = new kmeans_t;
    res->labels.resize(data->size);

    // Initialize centroids
    for (int i = 0; i < data->num_cluster; i++) {
        int index = kmeans_rand() % data->size;
        res->centroids.push_back(data->points[index]);
    }

    int iterations = 0;
    std::vector<std::vector<double>> new_centroids = res->centroids;
    Timer t;
    do {
        res->centroids = new_centroids;

        int i = 0;
        std::vector<int> num_points(data->num_cluster, 0);
        for (auto& p : data->points) {
            res->labels[i] = find_nearest_centroid(p, res->centroids);
            num_points[res->labels[i]]++;
            ++i;
        }
        // std::cout << "Num points: " << num_points << std::endl;
        // Initialize centroids at origin
        new_centroids = std::vector<std::vector<double>>(data->num_cluster, std::vector<double>(data->dims, 0.0));
        i = 0;
        for (auto& p : data->points) {

            auto& centroid = new_centroids[res->labels[i]];
            std::transform(centroid.begin(), centroid.end(), p.begin(), centroid.begin(), std::plus<double>());
            i++;
        }
        i = 0;
        for (auto& centroid : new_centroids) {
            std::transform(centroid.begin(), centroid.end(), centroid.begin(), [&](double a){return num_points[i] == 0 ? 0.0 : a / num_points[i];});
            i++;
        }
    } while(iterations++ < args->max_num_iter && !converged(res->centroids, new_centroids, args->threshold));
    std::cout << iterations << "," << t.elapsed() / iterations << std::endl;
    res->centroids = new_centroids;
    return res;
}

int main(int argc, char *argv[]) {
    args_t* args = parse_arguments(argc, argv);
    dataset_t* dataset = load_dataset(args);
    kmeans_srand(args->seed);

    kmeans_t* res = kmeans(dataset, args);

    if (args->output_centroids) {
        for (int clusterId = 0; clusterId < dataset->num_cluster; clusterId ++){
            std::cout << clusterId << " ";
            for (int d = 0; d < dataset->dims; d++)
                std::cout << res->centroids[clusterId][d] << " ";
            std::cout << std::endl;
        }
    } else {
        std::cout << "clusters:";
        for (int idx = 0; idx < dataset->size; idx ++) {
            std::cout << " " << res->labels[idx];
        }
    }
    return 0;
}