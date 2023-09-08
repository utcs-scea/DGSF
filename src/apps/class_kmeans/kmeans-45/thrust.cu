#include <random>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sort.h>
#include "./lib/argparse.h"
#include "./lib/datasets.h"
#include "./lib/thrust_utils.h"
#include "./lib/timer.h"

using namespace thrust::placeholders;


static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

struct kmeans_t {
    thrust::host_vector<double> last_centroids;
    thrust::host_vector<double> centroids;
    thrust::host_vector<int> labels;
};

struct argmin : public thrust::binary_function<thrust::tuple<int, double>, thrust::tuple<int, double>, thrust::tuple<int, double>>
{
    __host__ __device__
        thrust::tuple<int, double> operator()(const thrust::tuple<int, double>& a, const thrust::tuple<int, double>& b) const
    {
        if (thrust::get<1>(a) < thrust::get<1>(b)) {
            return a;
        }
        else {
            return b;
        }
    }
};

struct tuplesum : public thrust::binary_function<thrust::tuple<double, int>, thrust::tuple<double, int>, thrust::tuple<double, int>>
{
    __host__ __device__
        thrust::tuple<double, int> operator()(const thrust::tuple<double, int>& a, const thrust::tuple<double, int >& b) const
    {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(a) + thrust::get<1>(b));
    }
};

struct divide_tuple : public thrust::unary_function<thrust::tuple<double, int>, double>
{
    __host__ __device__
        double operator()(const thrust::tuple<double, int>& a) const
    {
        return thrust::get<0>(a) / (double) thrust::get<1>(a);
    }
};

struct extract_tuple : public thrust::unary_function<thrust::tuple<int, double>, int>
{
    __host__ __device__
        int operator()(const thrust::tuple<int, double>& a) const
    {
        return thrust::get<0>(a);
    }
};

void iterate(kmeans_t& result, const thrust::device_vector<double>& d_points, int dims, int size, int k) {
    thrust::device_vector<double> d_dists(size * k);
    thrust::device_vector<double> d_indices(size * dims);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::transform(d_indices.begin(), d_indices.end(), thrust::make_constant_iterator(dims), d_indices.begin(),
        thrust::divides<int>());

    thrust::device_vector<double> d_centroids = result.centroids;
    
    // Iteratively calculate distances from all points to each centroid
    for (int i = 0; i < k; i++) {
        // Create a tiled iterator of the centroid we are interested in for each point, this iterator is the size of the d_points vector
        tiled_range<thrust::device_vector<double>::iterator> centroid(d_centroids.begin() + (dims * i), d_centroids.begin() + (dims * (i + 1)), size);

        // Calculate squared diff between each element
       thrust::device_vector<double> d_expanded(size * dims);
       thrust::transform(d_points.begin(), d_points.end(), centroid.begin(), d_expanded.begin(), ((_1 - _2) * (_1 - _2)));
       thrust::device_vector<int> d_order(size);
       thrust::device_vector<double> d_dist(size);
       thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_expanded.begin(), d_order.begin(), d_dist.begin());
       // Scatter into empty slots in the d_dists vector
       thrust::scatter(d_dist.begin(), d_dist.end(), skipping_iterator(k).begin(), d_dists.begin() + i);
    }

    // Calculate the labels
    thrust::device_vector<thrust::tuple<int, double>> min_dists(size);
    thrust::device_vector<int> keys_out(size);
    thrust::device_vector<double> d_dist_indices(size * k);
    // Create indice list 1 .. 1 (k time) 2 .. 2 (k times) for use in reduction
    thrust::sequence(d_dist_indices.begin(), d_dist_indices.end());
    thrust::transform(d_dist_indices.begin(), d_dist_indices.end(), thrust::make_constant_iterator(k), d_dist_indices.begin(),
        thrust::divides<int>());
    thrust::reduce_by_key(
        d_dist_indices.begin(), d_dist_indices.end(),
        // Zip distances with their column number which corresponds to the id of the centroid
        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_transform_iterator(thrust::make_counting_iterator(0), _1 % k), d_dists.begin())),
        keys_out.begin(), min_dists.begin(),
        thrust::equal_to<int>(),
        argmin()
        );
    thrust::device_vector<int> labels(size);
    thrust::transform(min_dists.begin(), min_dists.end(), labels.begin(), extract_tuple());

    // Recalculate centroids from labels
    // Key all points with their centroid (repeat each label dims times)
    thrust::device_vector<int> ckeys(size * dims);
    thrust::constant_iterator<int> dims_it(dims);
    expand(dims_it, dims_it + size, labels.begin(), ckeys.begin());
    // Transform keys to unique key per centroid column (k * dims + seq % dims)
    thrust::transform(ckeys.begin(), ckeys.end(), thrust::make_counting_iterator(0), ckeys.begin(), _1 * dims + (_2 % dims));
    // Sum and count per key, transform to yield average
    thrust::device_vector<int> sum_keys(k * dims);
    thrust::device_vector<thrust::tuple<double, int>> sum_points(k * dims);


    auto it = thrust::make_zip_iterator(thrust::make_tuple(d_points.begin(), thrust::make_constant_iterator(1)));
    thrust::device_vector<thrust::tuple<double, int>> d_sorteddata(it, it + (size * dims));
    thrust::sort_by_key(ckeys.begin(), ckeys.end(), d_sorteddata.begin());

    thrust::reduce_by_key(
        ckeys.begin(), ckeys.end(),
        d_sorteddata.begin(),
        sum_keys.begin(),
        sum_points.begin(),
        thrust::equal_to<int>(),
        tuplesum()
    );
    thrust::device_vector<double> new_centroids(k * dims);
    thrust::transform(sum_points.begin(), sum_points.end(), new_centroids.begin(), divide_tuple());
    // Sort by keys to yield new centroids
    thrust::sort_by_key(sum_keys.begin(), sum_keys.end(), new_centroids.begin());

    result.labels = labels;
    result.last_centroids = result.centroids;
    result.centroids = new_centroids;
}

bool converged(const thrust::host_vector<double>& a, const thrust::host_vector<double>&  b, double threshold, int num_cluster, int dims) {
    for (int i = 0; i < num_cluster; ++i) {
        double dist = 0.0;
        for (int j = 0; j < dims; j++) {
            double x = a[i * dims + j] - b[i * dims + j];
            dist += x * x;
        }
        // Compare squared distance
        if (dist > threshold * threshold) {
            return false;
        }
    }
    return true;
}

kmeans_t* kmeans(cdataset_t* data, args_t* args) {
    kmeans_t* res = new kmeans_t;
    res->labels.resize(data->size);

    thrust::device_vector<double> d_points(data->points);

    // Initialize centroids
    std::vector<double> init_centroids(data->num_cluster * data->dims);
    for (int i = 0; i < data->num_cluster; i++) {
        int index = kmeans_rand() % data->size;
        std::copy(data->points.cbegin() + (index * data->dims), data->points.cbegin() + ((index + 1) * data->dims), init_centroids.begin() + (i * data->dims));
    }
    res->centroids = init_centroids;

    int iterations = 0;

    Timer t;
    do {
        iterate(*res, d_points, data->dims, data->size, data->num_cluster);
    } while (iterations++ < args->max_num_iter && !converged(res->last_centroids, res->centroids, args->threshold, data->num_cluster, data->dims));
    std::cout << iterations << "," << t.elapsed() / iterations << std::endl;
    return res;
}

int main(int argc, char* argv[]) {
    args_t* args = parse_arguments(argc, argv);
    cdataset_t* dataset = load_cdataset(args);
    kmeans_srand(args->seed);
    kmeans_t* res = kmeans(dataset, args);

    if (args->output_centroids) {
        for (int clusterId = 0; clusterId < dataset->num_cluster; clusterId++) {
            std::cout << clusterId << " ";
            for (int d = 0; d < dataset->dims; d++)
                std::cout << res->centroids[clusterId * dataset->dims + d] << " ";
            std::cout << std::endl;
        }
    }
    else {
        std::cout << "clusters:";
        for (int idx = 0; idx < dataset->size; idx ++) {
            std::cout << " " << res->labels[idx];
        }
    }
    return 0;
}
