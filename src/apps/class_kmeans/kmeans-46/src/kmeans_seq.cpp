#include "kmeans_seq.h"

#define SQ(v) ((v) * (v))

float calc_sq_dist(int     num_dims, 
                   float* i1, 
                   float* i2) {
    float sq_dist = 0; 

    for (int d = 0; d < num_dims; d++) {
        sq_dist += SQ(i1[d] - i2[d]);
    }
    return sq_dist;
}

int find_min_dist_centroid(struct options_t* opts,
                           int               data_idx, 
                           float*            data,
                           float*            curr_centroids) {
    float min_sq_dist = std::numeric_limits<float>::max();
    int min_cluster    = 0;
    for (int i = 0; i < opts->k_clusters; i++) {
        float* data_pt        = data + (data_idx * opts->n_dims);
        float* centroid_pt    = curr_centroids + (i * opts->n_dims);
        float cluster_sq_dist = calc_sq_dist(opts->n_dims, data_pt, centroid_pt);
     
        if (cluster_sq_dist < min_sq_dist) { 
            min_sq_dist = cluster_sq_dist; 
            min_cluster = i;
        }
    }
    return min_cluster;
}

void add_point_to_centroid(float*            new_centroid_sums, 
                           int*              new_centroid_cnts,
                           struct options_t* opts, 
                           int               data_idx, 
                           float*            data, 
                           float*            curr_centroids) { 
    int min_cluster = find_min_dist_centroid(opts, data_idx, data, curr_centroids);
    for (int i = 0; i < opts->n_dims; i++) {     
        new_centroid_sums[(min_cluster * opts->n_dims) + i] += data[(data_idx * opts->n_dims) + i];
    }
    new_centroid_cnts[min_cluster]++;
}

void avg_centroid_points(float*           centroids,
                         struct options_t* opts,
                         int               cluster_idx,
                         int*              centroid_cnts) {
    for (int i = 0; i < opts->n_dims; i++) {
        centroids[(cluster_idx * opts->n_dims) + i] /= centroid_cnts[cluster_idx];
    }
}

int check_convergence(struct options_t* opts,
                      float*           old_centroids,
                      float*           new_centroids) { 
    for (int i = 0; i < opts->k_clusters; i++) {
        float* old_centroid_pt = old_centroids + (opts->n_dims * i);
        float* new_centroid_pt = new_centroids + (opts->n_dims * i);         
        float sq_dist          = calc_sq_dist(opts->n_dims, old_centroid_pt, new_centroid_pt);
        if (sq_dist > opts->threshold) {
            return false;
        }
    }
    return true;
}

void kmeans_seq(float**          centroids_p, 
                int*              iterations_p, 
                int*              exec_milliseconds_p, 
                struct options_t* opts, 
                float*           input_vals, 
                int               n_vals) {  
    
	int num_centroids_bytes    = opts->k_clusters * opts->n_dims * sizeof(float);
	int num_centroid_cnt_bytes = opts->k_clusters * sizeof(int);
	
    float* curr_centroids  = *centroids_p;
    float* new_centroids   = (float*)malloc(num_centroids_bytes);
    int* new_centroid_cnts = (int*)malloc(num_centroid_cnt_bytes);
    memset(new_centroids, 0, num_centroids_bytes);
    memset(new_centroid_cnts, 0, num_centroid_cnt_bytes);
    
    int exec_milliseconds = 0;
    int iterations        = 0;
    bool done             = false;
                  
    while (!done) {
        auto exec_start = std::chrono::high_resolution_clock::now();    
        for (int i = 0; i < n_vals; i++) { 
            add_point_to_centroid(new_centroids, new_centroid_cnts, opts, i, input_vals, curr_centroids);
        }
        for (int i = 0; i < opts->k_clusters; i++) {
            avg_centroid_points(new_centroids, opts, i, new_centroid_cnts);
        }
        
        iterations++;
        if (iterations >= opts->max_num_iters || check_convergence(opts, curr_centroids, new_centroids)) {
            done = true;
        }
        
        float* temp_centroids = curr_centroids;
        curr_centroids = new_centroids;
        new_centroids  = temp_centroids;
            
        memset(new_centroids,     0, num_centroids_bytes);
        memset(new_centroid_cnts, 0, num_centroid_cnt_bytes);
        auto exec_end = std::chrono::high_resolution_clock::now();
        auto exec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_start);   
        
        exec_milliseconds += exec_diff.count(); 
    }
    *centroids_p         = curr_centroids;
    *exec_milliseconds_p = exec_milliseconds;
    *iterations_p        = iterations;
}