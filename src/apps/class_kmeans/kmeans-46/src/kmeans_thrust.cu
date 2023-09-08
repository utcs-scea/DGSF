#include "kmeans_thrust.h"

#define SQ(v) ((v) * (v))

__host__ __device__ 
    float calc_sq_dist(int     num_dims, 
                       const float* i1, 
                       const float* i2) {
    float sq_dist = 0; 

    for (int d = 0; d < num_dims; d++) {
        sq_dist += SQ(i1[d] - i2[d]);
    }
    return sq_dist;
}

struct get_pt_centroid_dist_func {
    const float* _m_pts;
    const float* _m_centroids;
    float* _m_dists;
    int _n_vals;
    int _k_clusters;
    int _n_dims;
    
    get_pt_centroid_dist_func(thrust::device_vector<float> const& pts, 
                              thrust::device_vector<float> const& centroids, 
                              thrust::device_vector<float>& dists,
                              int n_dims) {
        _m_pts       = thrust::raw_pointer_cast(pts.data());
        _m_centroids = thrust::raw_pointer_cast(centroids.data());
        _m_dists     = thrust::raw_pointer_cast(dists.data());
        _n_vals      = pts.size() / n_dims;
        _n_dims      = n_dims;
    }

    __host__ __device__ void operator()(int idx) const {
        int pt_idx = idx % _n_vals;
        int c_idx  = idx / _n_vals;
            
        const float* data_pt        = _m_pts + (pt_idx * _n_dims);
        const float* centroid_pt    = _m_centroids + (c_idx * _n_dims);
        float cluster_sq_dist = calc_sq_dist(_n_dims, data_pt, centroid_pt);
        _m_dists[(c_idx * _n_vals) + pt_idx] = cluster_sq_dist;
    }
};

void kmeans_thrust(float**           centroids_p, 
				   int*              iterations_p, 
				   int*              copy_milliseconds_p,
				   int*              exec_milliseconds_p, 
				   struct options_t* opts, 
				   float*            input_vals, 
				   int               n_vals) {
    
    int num_input_vals   = n_vals * opts->n_dims;
    int num_cluster_vals = opts->k_clusters * opts->n_dims;
    int num_c_pt_pairs   = n_vals * opts->k_clusters;
    int num_c_pt_vals    = num_c_pt_pairs * opts->n_dims;
    
    float* centroids = *centroids_p;
        
    // 0, 1, ... k, 0, 1, ... k ... 0, 1, ... k (n_vals tuples)
    auto data_idx_by_centroid_iter_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0), thrust::placeholders::_1 % n_vals);
    auto data_idx_by_centroid_iter_end   = data_idx_by_centroid_iter_begin + num_c_pt_pairs;  
        
    // 0, 0, ... 0 (k_vals), 1, 1, ... 1 (k_vals) ... n, n, ... n (k_vals)
    auto centroid_idx_by_data_iter_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0), thrust::placeholders::_1 / n_vals);
    auto centroid_idx_by_data_iter_end   = centroid_idx_by_data_iter_begin + num_c_pt_pairs;
    
    thrust::device_vector<float> d_input_vals(input_vals, input_vals + num_input_vals);
    thrust::device_vector<float> d_curr_centroids(centroids, centroids + num_cluster_vals);
    thrust::device_vector<float> d_pt_centroid_dists(num_c_pt_vals, 0.0);
    thrust::device_vector<float> d_new_centroids(num_cluster_vals, 0.0);
    thrust::device_vector<int>   d_new_centroid_cnts(opts->k_clusters, 0);

    // 
    thrust::device_vector<int>   d_min_centroid_dists(n_vals, 0.0);
    thrust::device_vector<int>   d_min_centroid_dist_idxs(n_vals, 0);

    bool done = false;  
    while (!done) { 
        // find the distance between each point and each centroid
        // dist(p0, c0), ..., dist(p0, ck), dist(p1, c0), ... dist(p1, ck), ... dist(pn, c0), ... dist(pn, ck)
        thrust::for_each_n(thrust::counting_iterator<size_t>(0), num_c_pt_pairs, 
                           get_pt_centroid_dist_func(d_input_vals, d_curr_centroids, d_pt_centroid_dists, opts->n_dims));

        // find the minimum centroid distance for each point
        // dist(p0, c_min_dist), dist(p1, c_min_dist), ... dist(pn, c_min_dist)
        thrust::reduce_by_key(data_idx_by_centroid_iter_begin, 
                              data_idx_by_centroid_iter_end,
                              // (dist, data_idx)
                              thrust:make_zip_iterator(thrust::make_tuple(d_pt_centroid_dists, centroid_idx_by_data_iter_begin)),
                              thrust::make_discard_iterator(), 
                              // (min_dist, centroid_idx)
                              thrust::make_zip_iterator(thrust::make_tuple(d_min_centroid_dists.begin(), d_min_centroid_dist_idxs.begin())), 
                              thrust::equal_to<float>(), 
                              thrust::minimum<thrust::tuple<float, float> >());

        // sum the points that are the closest for each centroid 
        
        
        // sum the no. points that are mapped to each centroid (count)
        
        // average by the no. points in each centroid bucket
        
        // check for convergence
        
    }
}
