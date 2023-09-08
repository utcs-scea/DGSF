#include "kmeans_init.h"

static unsigned long int _next   = 1;
static unsigned long _kmeans_rmax = 32767;  

void kmeans_srand(unsigned int seed) {
    _next = seed;
}

int kmeans_rand() {
    _next = _next * 1103515245 + 12345;
    return (unsigned int)(_next/65536) % (_kmeans_rmax+1);
}

float* init_centroids(struct options_t* opts, 
                       int              n_vals, 
                       float*           data) { 
    float* curr_centroids = (float*)malloc(opts->k_clusters * opts->n_dims * sizeof(float));
    kmeans_srand(opts->seed);
    for (int i = 0; i < opts->k_clusters; i++){
        int centroid_idx = kmeans_rand() % n_vals;
        for (int j = 0; j < opts->n_dims; j++) {
            curr_centroids[(i * opts->n_dims) + j] = data[(centroid_idx * opts->n_dims) + j];
        }
    }
    return curr_centroids;
}