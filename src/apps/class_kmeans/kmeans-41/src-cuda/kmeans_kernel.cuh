__global__ void dot(double* points, double* centroids, int dims, int npoints, int ncentroids, int points_size, int centroids_size, int cross_size, double* distances);
__global__ void sqrt_kernel(double* values, int n);
__global__ void nearest_centroid(double* distances, int npoints, int ncentroids, int cross_size, int* nearest_centroids);
__global__ void count_centroid_id(int* nearest_centroids, int npoints, int ncentroids, int* counts);
__global__ void sum_new_centroid_values(double* points, int* nearest_centroids, int dims, int npoints, int points_size, int centroids_size, double* new_centroid_values);
__global__ void avg_new_centroid_values(double* new_centroid_values, int* counts, int dims, int ncentroids, int centroids_size);
__global__ void new_centroid_movement_squared(double* centroids, double* new_centroids, int dims, int ncentroids, int centroids_size, double* distances);
__global__ void is_convergent(double* distances, double threshold, int ncentroids, int* ret);
__global__ void reset_zero(double* array, int n);

__global__ void nearest_centroid_shared(double* points, double* centroids, int dims, int npoints, int ncentroids, int points_size, int centroids_size, int cross_size, int* nearest_centroids);