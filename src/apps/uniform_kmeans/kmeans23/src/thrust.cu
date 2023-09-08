#include <chrono>
#include <cmath>
#include <iostream>
#include <functional>
#include <limits>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "common.h"


struct euclidean_distance_squared_functor {
    const float *mPoints;
    const float *mCentroids;
    const int mDims;
    const int mNum_cluster;

    euclidean_distance_squared_functor(
        const float *points,
        const float *centroids,
        int dims,
        int num_cluster) :
        mPoints(points),
        mCentroids(centroids),
        mDims(dims),
        mNum_cluster(num_cluster)
    {
    }

    __host__ __device__
    float operator()(int distance_idx)
    {
        int point_id = distance_idx / mNum_cluster;
        int centroid_id = distance_idx % mNum_cluster;

        double dist = 0.0;
        for (int dim = 0; dim < mDims; ++dim) {

            double diff =
                mPoints[point_id * mDims + dim] -
                mCentroids[centroid_id * mDims + dim];

            dist += diff * diff;
        }
        return dist;
    }
};

struct min_distance_functor {
    const float *mDistances;
    const int mNum_cluster;

    min_distance_functor(
        const float *distances,
        int num_cluster) :
        mDistances(distances),
        mNum_cluster(num_cluster)
    {
    }

    __host__ __device__
    int operator()(int point_id)
    {
        int start_idx = point_id * mNum_cluster;
        float min_dist = 999999999;
        int label = -1;

        for (int i = start_idx; i < start_idx + mNum_cluster; ++i) {
            if (mDistances[i] < min_dist) {
                label = i % mNum_cluster;
                min_dist = mDistances[i];
            }
        }
        return label;
    }
};

void find_nearest_centroids(
    const thrust::device_vector<float> &dev_points, 
    const thrust::device_vector<float> &dev_centroids,
    thrust::device_vector<int> &dev_labels,
    const options_t &opts)
{
    thrust::device_vector<float> distances(
        dev_points.size() / opts.dims * opts.num_cluster);

    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(
            dev_points.size() / opts.dims * opts.num_cluster),
        distances.begin(),
        euclidean_distance_squared_functor(
            dev_points.data().get(),
            dev_centroids.data().get(),
            opts.dims,
            opts.num_cluster
        )
    );

    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(dev_points.size() / opts.dims),
        dev_labels.begin(),
        min_distance_functor(distances.data().get(), opts.num_cluster)
    );
}

struct update_centroid_functor {
    const float *mPoints;
    const int *mLabels;
    const int mDims;
    const int mLabelsSize;

    update_centroid_functor(
        const float *points,
        const int *labels,
        int dims,
        int labelSize) :
        mPoints(points),
        mLabels(labels),
        mDims(dims),
        mLabelsSize(labelSize)
    {
    }

    __host__ __device__
    float operator()(int centroid_idx) {
        int centroid_id = centroid_idx / mDims;
        int dim = centroid_idx % mDims;
        int count = 0;
        float newVal = 0.0;

        for (int i = 0; i < mLabelsSize; ++i) {
            if (mLabels[i] == centroid_id) {
                ++count;
                newVal += mPoints[i * mDims + dim];
            }
        }

        return newVal / count;
    }
};

void update_centroids(
    const thrust::device_vector<float> &dev_points,
    thrust::device_vector<float> &dev_centroids,
    const thrust::device_vector<int> &dev_labels,
    const options_t &opts)
{
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(dev_centroids.size()),
        dev_centroids.begin(),
        update_centroid_functor(
            dev_points.data().get(),
            dev_labels.data().get(),
            opts.dims,
            dev_labels.size()
        )
    );
}

struct convergence_functor {
    const float *mOldCentroids;
    const float *mCentroids;
    const int mDims;
    const int mNum_cluster;
    const float mThreshold;

    convergence_functor(
        const float *oldCentroids,
        const float *centroids,
        int dims,
        int num_cluster,
        float threshold) :
        mOldCentroids(oldCentroids),
        mCentroids(centroids),
        mDims(dims),
        mNum_cluster(num_cluster),
        mThreshold(threshold)
    {
    }

    __host__ __device__
    bool operator()(int centroid_id)
    {
        double dist = 0.0;
        for (int dim = 0; dim < mDims; ++dim) {

            double diff = 
                mOldCentroids[centroid_id * mDims + dim] -
                mCentroids[centroid_id * mDims + dim];

            dist += diff * diff;
        }

        return sqrt(dist) < mThreshold;
    }
};

bool converged(
    const thrust::device_vector<float> &old_centroids,
    const thrust::device_vector<float> &centroids,
    const options_t &opts)
{
    thrust::device_vector<float> converged(centroids.size() / opts.dims);

    thrust::transform( 
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(centroids.size() / opts.dims),
        converged.begin(),
        convergence_functor(
            old_centroids.data().get(),
            centroids.data().get(),
            opts.dims,
            opts.num_cluster,
            opts.threshold
        )
    );

    return thrust::all_of(
        converged.begin(),
        converged.end(),
        thrust::identity<bool>());
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

    thrust::device_vector<float> dev_points(points);
    thrust::device_vector<float> dev_centroids(centroids);
    thrust::device_vector<float> dev_old_centroids(centroids.size());
    thrust::device_vector<int> dev_labels(n_points);

	// Start -> Stop Events used to record time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int i = 0;
    while (++i < opts.max_num_iter) {
        thrust::copy(
            dev_centroids.begin(),
            dev_centroids.end(),
            dev_old_centroids.begin());

        find_nearest_centroids(dev_points, dev_centroids, dev_labels, opts);
        update_centroids(dev_points, dev_centroids, dev_labels, opts);
        bool done = converged(dev_old_centroids, dev_centroids, opts);

        if (done) {
            break;
        }
    }

    //End timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time_in_ms = 0;
    cudaEventElapsedTime(&total_time_in_ms, start, stop);

    thrust::copy(dev_points.begin(), dev_points.end(), points.begin());
    thrust::copy(dev_centroids.begin(), dev_centroids.end(), centroids.begin());
    std::vector<int> labels(n_points);
    thrust::copy(dev_labels.begin(), dev_labels.end(), labels.begin());

    output_results(
        labels,
        centroids,
        total_time_in_ms,
        i,
        opts);

    return 0;
}