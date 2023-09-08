#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include "loader.h"
#include "argparse.h"
#include "rng.h"

struct calculate_distance {
    double* points_it;
    double* centroids_it;
    int dims;
    int ncentroids;
    
    calculate_distance(
        double* _points_it,
        double* _centroids_it,
        int _dims,
        int _ncentroids
    ): points_it(_points_it), centroids_it(_centroids_it), dims(_dims), ncentroids(_ncentroids) {}
    
    __host__ __device__ double operator()(const int& i) const {
        int point_idx = i / ncentroids;
        int centroid_idx = i % ncentroids;
        double sum = 0;
        for (int d = 0; d < dims; d++) {
            double diff = points_it[point_idx * dims + d] - centroids_it[centroid_idx * dims + d];
            sum += diff * diff;
        }
        return sum;
    }
};

struct times_n {
    int n;
    times_n(int _n): n(_n) {}
    __host__ __device__ inline int operator()(const int& x) const {
        return x * n;
    }
};

struct divide_by_n {
    int n;
    divide_by_n(int _n): n(_n) {}
    __host__ __device__ inline int operator()(const int& x) const {
        return x / n;
    }
};

struct modulo_n {
    int n;
    modulo_n(int _n): n(_n) {}
    __host__ __device__ inline int operator()(const int& x) const {
        return x % n;
    }
};

struct centroid_value_idx {
    int _dims;
    int* _centroids_idx;
    
    centroid_value_idx(int* centroids_idx, int dims): _centroids_idx(centroids_idx), _dims(dims) {}
    __host__ __device__ int operator()(const int& i) const {
        int point_idx = i / _dims;
        return i % _dims + _centroids_idx[point_idx] * _dims;
    }
};

struct centroid_idx {
    int _dims;
    int* _centroids_idx;
    
    centroid_idx(int* centroids_idx, int dims): _centroids_idx(centroids_idx), _dims(dims) {}
    __host__ __device__ int operator()(const int& i) const {
        int point_idx = i / _dims;
        return _centroids_idx[point_idx];
    }
};

struct avg_centroid {
    int _dims;
    int* _centroid_count;
    
    avg_centroid(int* centroid_count, int dims): _centroid_count(centroid_count), _dims(dims) {}
    __host__ __device__ double operator()(const double& v, const int& i) const {
        int centroid_idx = i / _dims;
        return v / _centroid_count[centroid_idx];
    }
};

struct convergence_test : public thrust::unary_function<int, bool> {
    double* centroids_it;
    double* new_centroids_it;
    int dims;
    double threshold;
    
    convergence_test(
        double* _centroids_it,
        double* _new_centroids_it,
        int _dims,
        double _threshold
    ): centroids_it(_centroids_it), new_centroids_it(_new_centroids_it), dims(_dims), threshold(_threshold) {}
    
    __host__ __device__ bool operator()(const int& i) const {
        double sum = 0;
        for (int d = 0; d < dims; d++) {
            double diff = new_centroids_it[i * dims + d] - centroids_it[i * dims + d];
            sum += diff * diff;
        }
        return sqrt(sum) < threshold;
    }
};

int main(int argc, char** argv) {
    options_t opts;
    get_opts(argc, argv, &opts);
    loader file_loader(opts.dims);
    
    int dims = opts.dims;
    std::vector<double> p = file_loader.load_as_1d(opts.inputfilename);
    int npoints = p.size() / dims;
    int ncentroids = opts.num_cluster;
    
    thrust::device_vector<double> points(p.begin(), p.end()), centroids(dims * opts.num_cluster), new_centroids(dims * opts.num_cluster), distances(npoints * ncentroids), minimum_distance(npoints), tpoints(p.size());
    thrust::device_vector<int> minimum_centroid_id(npoints), tid(npoints), new_centroid_count(ncentroids), x(p.size());
    
    rng randomizer(opts.seed);
    for (int i = 0; i < opts.num_cluster; i++) {
        int idx = randomizer.kmeans_rand() % npoints;
        for (int j = 0; j < dims; j++) {
            centroids[i * dims + j] = points[idx * dims + j];
        }
    }
    
    bool is_convergent = false;
    int iter = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    do {
        thrust::transform(
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(npoints * ncentroids),
            distances.begin(),
            calculate_distance(
                thrust::raw_pointer_cast(points.data()),
                thrust::raw_pointer_cast(centroids.data()),
                dims,
                ncentroids
            )
        );

        auto key_it = thrust::make_transform_iterator(thrust::make_counting_iterator(0), divide_by_n(ncentroids));
        thrust::reduce_by_key(
            key_it,
            key_it + (ncentroids * npoints),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    distances.begin(),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        modulo_n(ncentroids)
                    )
                )
            ),
            thrust::make_discard_iterator(),
            thrust::make_zip_iterator(thrust::make_tuple(minimum_distance.begin(), minimum_centroid_id.begin())),
            thrust::equal_to<int>(),
            thrust::minimum<thrust::tuple<double, int>>()
        );
        
        auto point_value_idx_modified_it = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            centroid_value_idx(thrust::raw_pointer_cast(minimum_centroid_id.data()), dims)
        );
        
        thrust::copy(point_value_idx_modified_it, point_value_idx_modified_it + (dims * npoints), x.begin());
        thrust::copy(points.begin(), points.end(), tpoints.begin());
        thrust::sort_by_key(
            x.begin(),
            x.end(),
            tpoints.begin()
        );

        thrust::reduce_by_key(
            x.begin(),
            x.end(),
            tpoints.begin(),
            thrust::make_discard_iterator(),
            new_centroids.begin(),
            thrust::equal_to<int>()
        );
        
        thrust::copy(minimum_centroid_id.begin(), minimum_centroid_id.end(), tid.begin());
        thrust::sort(tid.begin(), tid.end());
        thrust::reduce_by_key(
            tid.begin(), 
            tid.end(), 
            thrust::make_constant_iterator(1), 
            thrust::make_discard_iterator(),
            new_centroid_count.begin()
        );

        thrust::transform(
            new_centroids.begin(), 
            new_centroids.end(),
            thrust::make_counting_iterator(0),
            new_centroids.begin(), 
            avg_centroid(thrust::raw_pointer_cast(new_centroid_count.data()), dims)
        );

        auto new_centroid_distance_iterator = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            convergence_test(
                thrust::raw_pointer_cast(centroids.data()),
                thrust::raw_pointer_cast(new_centroids.data()),
                dims,
                opts.threshold
            )
        );

        is_convergent = thrust::all_of(
            new_centroid_distance_iterator,
            new_centroid_distance_iterator + ncentroids,
            thrust::identity<bool>()
        );

        centroids = new_centroids;
        iter++;
    } while (!is_convergent && iter < opts.max_num_iter);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("%d,%ld\n", iter, diff.count());
    if (opts.output_centroids) {
        std::vector<double> vcentroids(ncentroids * dims);
        thrust::copy(centroids.begin(), centroids.end(), vcentroids.begin());
        for (int i = 0; i < ncentroids; i++) {
            printf("%d ", i);
            for (int j = 0; j < dims; j++) {
                printf("%lf ", vcentroids[j + dims * i]);
            }
            printf("\n");
        }
    } else {
        std::vector<int> vcluster(npoints);
        thrust::copy(minimum_centroid_id.begin(), minimum_centroid_id.end(), vcluster.begin());
        printf("clusters:");
        for (int i = 0; i < npoints; i++) {
            printf(" %d", vcluster[i]);
        }
    }
    
    return 0;
}