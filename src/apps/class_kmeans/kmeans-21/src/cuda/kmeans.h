#ifndef _KMEANS_H
#define _KMEANS_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);															   \
    }                                                                          \
}

struct kmeans_iteration_t{
    bool converged;
    float time_taken;
};

kmeans_iteration_t kmeans_cuda(cudaDeviceProp deviceProp, options_t opts, int n_vals,float *d_points,float *d_centroids,int *d_centroid_assignments,  int *d_centroid_counts, float *d_point_centroid_distances, float **h_centroids,float **h_old_centroids);

#endif