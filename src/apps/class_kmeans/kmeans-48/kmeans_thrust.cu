#include <iostream>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <chrono>


// random centroid generation code
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}


struct nearest_centroid_operator {
    thrust::device_ptr<double> _points;
    thrust::device_ptr<double> _centroids;
    int _num_clusters;
    int _dims;
    thrust::device_ptr<int> _device_point_sum_metadata;
    
    __host__ __device__ nearest_centroid_operator(thrust::device_ptr<double> points, 
                                              thrust::device_ptr<double> centroids, 
                                              int num_clusters, int dims, 
                                              thrust::device_ptr<int> device_point_sum_metadata): 
        _points(points), _centroids(centroids), _num_clusters(num_clusters), _dims(dims), 
        _device_point_sum_metadata(device_point_sum_metadata) {
        
    }
    
    __host__ __device__ int operator()(int point_index) {
        thrust::device_ptr<double> point = _points + point_index*_dims;
        
        int min_centroid = -1;
        double min_distance = 0.0;
        for (int i=0; i<_num_clusters; i++) {
            double distance = 0.0;
            for (int j=0; j<_dims; j++) {
                distance += (*(point + j) - *(_centroids+i*_dims+j))
                    *(*(point + j) - *(_centroids+i*_dims+j));
            }
            if (distance < min_distance || min_centroid < 0) {
                min_distance = distance;
                min_centroid = i;
            }
        }
        for (int j=0; j<_dims; j++) {
            *(_device_point_sum_metadata + point_index*_dims + j) = (min_centroid*_dims + j);
        }
        return min_centroid;
    }
};


struct compute_centroid_operator {
    thrust::device_ptr<double> _point_sums;
    thrust::device_ptr<int> _label_count;
    int _dims;
    __host__ __device__ compute_centroid_operator(thrust::device_ptr<double> point_sum, 
                                         thrust::device_ptr<int> labels_count,
                                         int dims):
        _point_sums(point_sum) , _label_count(labels_count), _dims(dims){
        
    }
    
    __host__ __device__ int operator()(int index) {
        thrust::device_ptr<double> centroid_start = _point_sums + index*_dims;
        for (int i=0; i<_dims; i++) {
            *(centroid_start+i) = (*(centroid_start+i))/double(*(_label_count+index));
        }
        return 0;
    }
};

struct point_wise_distance_operator {
    thrust::device_ptr<double> _old_centroids;
    thrust::device_ptr<double> _centroids;
    int _dims;
    
    __host__ __device__ point_wise_distance_operator(thrust::device_ptr<double> old_centroids, 
                                                     thrust::device_ptr<double> centroids, int dims):
        _old_centroids(old_centroids), _centroids(centroids), _dims(dims) {
            
        }
    
    __host__ __device__ double operator()(int centroid_index) {
        double distance = 0.0;
        for (int j=0; j<_dims; j++) {
            distance += (*(_old_centroids + centroid_index*_dims + j) 
                         - *(_centroids + centroid_index*_dims + j))
                * (*(_old_centroids + centroid_index*_dims + j) 
                         - *(_centroids + centroid_index*_dims + j));
        }
        return distance;
    }
};


int main(int argc, char *argv[]) {
    // take the commandline options
    int num_clusters, dims, max_num_iters, seed;
    char* fileName;
    double threshold;
    bool outputCentroids = false;
    int c;
    while ((c = getopt(argc, argv, "k:d:i:m:t:s:c")) != -1) {
        switch(c) {
            case 'k':
                num_clusters = atoi((char*)optarg);
                break;
            case 'd':
                dims = atoi((char*)optarg);
                break;
            case 'i':
                fileName = (char*)optarg;
                break;
            case 'm':
                max_num_iters = atoi((char*)optarg);
                break;
            case 't':
                threshold = atof((char*)optarg)/double(10.0);
                threshold = threshold*threshold;
                break;
            case 's':
                seed = atoi((char*)optarg);
                break;
            case 'c':
                outputCentroids = true;
                break;
        }
    }
    
    FILE *fp = fopen(fileName, "r");
    int num_inputs;
    c = fscanf(fp, "%d", &num_inputs);
    
    // data for the coordinates of the points
    thrust::host_vector<double> host_points(num_inputs*dims);
    int point_pos;
    for (int i=0; i<num_inputs; i++) {
        c = fscanf(fp, "%d", &point_pos);
        for (int j=0; j<dims; j++) {
            c = fscanf(fp, "%lf", &(host_points[i*dims + j]));
        }
    }
    thrust::device_vector<double> device_points = host_points;
    
    // data for the initial centroids
    // also store the initial centroid coordinates
    kmeans_srand(seed);
    thrust::host_vector<double> host_centroids(num_clusters*dims);
    for (int i=0; i<num_clusters; i++) {
        int index = kmeans_rand() % num_inputs;
        for (int j=0; j<dims; j++) {
            host_centroids[i*dims + j] = host_points[index*dims + j];
        }
    }
    thrust::device_vector<double> device_centroids = host_centroids;
    thrust::device_vector<double> old_device_centroids(num_clusters*dims);
    
    // data for the labels and label_count
    thrust::device_vector<int> device_labels(num_inputs);
    thrust::device_vector<int> device_labels_count(num_clusters);
    
    // metadata for the main loop
    thrust::device_vector<int> device_point_sum_metadata(num_inputs*dims);
    thrust::device_vector<int> device_labels_count_metadata(num_inputs, 1);
    thrust::device_vector<double> device_points_copy(num_inputs*dims);
    thrust::device_vector<int> device_labels_copy(num_inputs);
    nearest_centroid_operator nearest_centroid_op(device_points.data(),
                                                  device_centroids.data(), 
                                                  num_clusters, dims, 
                                                  device_point_sum_metadata.data());
    compute_centroid_operator compute_centroid_op(device_centroids.data(), 
                                                  device_labels_count.data(), 
                                                  dims);
    point_wise_distance_operator point_wise_distance_op(old_device_centroids.data(), 
                                                        device_centroids.data(), dims);
    thrust::device_vector<double> point_wise_distances(num_clusters);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int iters = 0;
    int should_break = 0;
    while (iters < max_num_iters) {
        // copy the centroids to old centroids
        thrust::copy(thrust::device, device_centroids.begin(), device_centroids.end(), 
                     old_device_centroids.begin());
        
        // compute which centroid is nearest to each point and metadata for summing points
        thrust::tabulate(thrust::device, device_labels.begin(), device_labels.end(), 
                         nearest_centroid_op);
        
        // make a copy of device points that can be sorted and sort it for summing
        thrust::copy(thrust::device, device_points.begin(), device_points.end(), 
                     device_points_copy.begin());
        thrust::sort_by_key(thrust::device, device_point_sum_metadata.begin(), 
                            device_point_sum_metadata.end(), device_points_copy.begin());
        
        // sum the point which are assigned to each cluster and store the result in device_centroids
        thrust::reduce_by_key(thrust::device, device_point_sum_metadata.begin(), 
                              device_point_sum_metadata.end(), device_points_copy.begin(), 
                             device_point_sum_metadata.begin(), device_centroids.begin());
        
        // copy the labels and find the sum
        thrust::copy(thrust::device, device_labels.begin(), device_labels.end(), 
                     device_labels_copy.begin());
        thrust::sort(thrust::device, device_labels_copy.begin(), 
                     device_labels_copy.end());
        thrust::reduce_by_key(thrust::device, device_labels_copy.begin(), device_labels_copy.end(), 
                              device_labels_count_metadata.begin(), device_labels_copy.begin(), 
                              device_labels_count.begin());
        
        // find the final centroids
        // this function will overwrite the labels count to 0 and device point sum
        // by label count to get correct centroids
        thrust::tabulate(thrust::device, device_labels_count.begin(), 
                         device_labels_count.end(), compute_centroid_op);
        
        
        
        iters++;
        // find the convergence score and compare to threshold
        // break depending on the result of the comparison
        thrust::tabulate(thrust::device, point_wise_distances.begin(), 
                         point_wise_distances.end(), point_wise_distance_op);
        double convergence_score = thrust::reduce(thrust::device, 
                                                  point_wise_distances.begin(), 
                                                  point_wise_distances.end());
        if (convergence_score < threshold) {
            should_break +=1;
        } else {
            should_break = 0;
        }
        if (should_break > 3) {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count())/(double)iters;

    printf("%d,%lf\n", iters, time/1000.0);
    // print the required thing here
    if (outputCentroids) {
        // copy the centroids from device to host
        host_centroids = device_centroids;
        // print the centroids
        for (int i=0; i<num_clusters; i++) {
            printf("%d ", i);
            for (int j=0; j<dims; j++) {
                printf("%lf ", host_centroids[i*dims + j]);
            }
            printf("\n");
        }
    } else {
        // copy labels from device to host
        thrust::host_vector<int> host_labels = device_labels;
        // print the labels
        printf("clusters:");
        for (int p=0; p < num_inputs; p++)
            printf(" %d", host_labels[p]);
    }
}