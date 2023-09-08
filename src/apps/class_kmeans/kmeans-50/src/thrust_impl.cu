#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include "argparse.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}


struct point_add : public thrust::binary_function<double,double,double>
{

  using Base = thrust::binary_function<double,double,double>;
  int dims;
  double* sum_centroids;
  double* data_points;
  
  point_add(int dim) :
  Base(),
  dims(dim) {}
  
  __host__ __device__
  double operator()(double x,
                    double y) {
    return x + y;
  }
};



struct calcCentroid {
     __host__ __device__
     double operator()(const double val, const int total) const {
         if (total != 0) {
             return val/total;
         }
         
         return val;
     }
    
};

// create num_points * dims representation using using num_points
struct createFullSize {
    int *m_full_size;
    int *m_num_size;
    int dims;
    createFullSize(int* full_size,
                   int* num_size,
                   int dim) :
    m_full_size(full_size),
    m_num_size(num_size),
    dims(dim) {};    
    
    __host__ __device__
    int operator()(const int point_index) const {
       m_full_size[point_index] = m_num_size[point_index/dims]*dims + (point_index % dims);
       // return a junk value 
       return 0;
    }
};

struct checkConvergence {
    double *m_previousCentroid;
    double *m_newCentroid;
    double m_threshold;
    int dims;
    checkConvergence(double* previousCentroid,
                     double* newCentroid,
                     double threshold,
                     int dim) :
    m_previousCentroid(previousCentroid),
    m_newCentroid(newCentroid),
    m_threshold(threshold),
    dims(dim)
    {};
    
    
     __host__ __device__
    int operator()(const int point_index) const {
        double l2_norm =0;
        int real_index = point_index * dims;
        for (int i=0; i<dims; i++) {
            double diff = m_newCentroid[real_index + i] - m_previousCentroid[real_index + i];
            l2_norm += diff*diff;
        }
         
        if (l2_norm > m_threshold*m_threshold) {
            return 1; // not converged
        }
         
        return 0; // this point converged
    }
};

 struct getCentroidIndex {
     double* k_centroids;
     double* data_points;
     int num_cluster;
     int dim;
     getCentroidIndex(double* centroids,
           double* data_points,
           int n_cluster,
           int dim):
     k_centroids(centroids),
     data_points(data_points),
     num_cluster(n_cluster),
     dim(dim) {}
     
     __host__ __device__
     double getDistanceSq(const double* point1, const double* point2) const {
        double distance_sq = 0;
        for(int i=0; i < dim; i++) {
            distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
        }
    
        return distance_sq;
     }
     
     __host__ __device__
     int operator()(const int point_index) const {
        double *point = data_points + point_index*dim;
        int nearest_index = 0;
        double smallest_dist =  getDistanceSq(point, k_centroids + nearest_index);
         
         
        for(int i=1; i < num_cluster; i++) {
            double temp = getDistanceSq(point, k_centroids + i*dim);
            
            if (temp < smallest_dist) {
                nearest_index = i;
                smallest_dist = temp;
            }
        }
    
        return nearest_index;
     }
 };

void read_file(struct options_t& options,
               int*              n_vals,
               std::vector<double>& data_points,
               std::vector<double>& k_centroids,
               std::vector<int>&    clusterid_of_points) {

    
    std::ifstream in;
    in.open(options.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array  
    data_points = std::vector<double>(*n_vals * options.dims);
    
    
    // Read input vals
    for (int i = 0; i < *n_vals * options.dims; ++i) {
        if (i % options.dims == 0) {
            // unused index
            double temp;
            in >> temp;  
        }
        
            double temp;
            in >> temp;  
        
        data_points[i] = temp;
    }
    

	// k centroids array init 
    k_centroids = std::vector<double>(options.num_cluster * options.dims);    
 
	clusterid_of_points=std::vector<int>(*n_vals);
    
    kmeans_srand(options.seed);	
    for (int i=0; i< options.num_cluster; i++) {
        int index = kmeans_rand() % (*n_vals);
        for(int j=0; j < options.dims; j++) {
            k_centroids[i * options.dims + j] = data_points[index*options.dims + j];
        }
    }
}



int main(int argc, char** argv) {
    struct options_t options;
    get_opts(argc,
             argv,
             &options);
    int num_vals = 0;
    std::vector<double> host_data_points;
    std::vector<double> host_k_centroids;
    std::vector<int> host_clusterid_of_points;
    
   
    
    read_file(options,
              &num_vals,
              host_data_points,
              host_k_centroids,
              host_clusterid_of_points);

    thrust::device_vector<double> data_points(host_data_points);
    thrust::device_vector<double> k_centroids(host_k_centroids);
    thrust::device_vector<double> k_sum_centroids(options.num_cluster * options.dims);
    thrust::device_vector<int> clusterid_of_points(host_clusterid_of_points);
    thrust::device_vector<double> k_previous_centroids(options.num_cluster * options.dims);    
        
    

    thrust::device_vector<int> num_points_each_cluster = thrust::device_vector<int>(options.num_cluster * options.dims);
    thrust::device_vector<int> index_of_each_cluster = thrust::device_vector<int>(options.num_cluster * options.dims);
    
    thrust::equal_to<int> binary_pred;     
    point_add p_add(options.dims);
    
    thrust::device_vector<int>  my_sequence(num_vals);
    thrust::sequence(my_sequence.begin(), my_sequence.end(), 0);

    
    thrust::device_vector<int>  k_centroid_sequence(options.num_cluster);
    thrust::sequence(k_centroid_sequence.begin(), k_centroid_sequence.end(), 0);
    thrust::device_vector<int>  k_centroid_check_convergence(options.num_cluster);
    
    thrust::device_vector<int>  my_sequence_full_size(num_vals * options.dims);
    thrust::device_vector<int>  temp_my_sequence_full_size(num_vals * options.dims);
    
    thrust::sequence(my_sequence_full_size.begin(), my_sequence_full_size.end(), 0);
    
    thrust::device_vector<int> singles_full_size(num_vals * options.dims, 1);
    thrust::device_vector<int> clusterid_of_points_full_size(num_vals * options.dims);
    // for sorting values; doing this using zip iterator seems to hurt performance
    thrust::device_vector<int> clusterid_of_points_full_size2(num_vals * options.dims);    

    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start);
    int iter = 0;
    for(; iter < options.max_num_iter; iter++) {

        // getCentroidIndex
        thrust::transform(thrust::device,
                          my_sequence.begin(),
                          my_sequence.end(),
                          clusterid_of_points.begin(),
                          getCentroidIndex(thrust::raw_pointer_cast(&k_centroids[0]),
                                 thrust::raw_pointer_cast(&data_points[0]),
                                 options.num_cluster,
                                 options.dims));
        

        // clusterid_of_points_full_size & 2 are filled with values; to be used for sort later
        thrust::transform(thrust::device,
                          my_sequence_full_size.begin(),
                          my_sequence_full_size.end(),
                          temp_my_sequence_full_size.begin(),
                          createFullSize(thrust::raw_pointer_cast(&clusterid_of_points_full_size[0]),
                                         thrust::raw_pointer_cast(&clusterid_of_points[0]),
                                         options.dims));
        
        thrust::transform(thrust::device,
                          my_sequence_full_size.begin(),
                          my_sequence_full_size.end(),
                          temp_my_sequence_full_size.begin(),
                          createFullSize(thrust::raw_pointer_cast(&clusterid_of_points_full_size2[0]),
                                         thrust::raw_pointer_cast(&clusterid_of_points[0]),
                                         options.dims));
                          
        
        // sort the data points
        thrust::stable_sort_by_key(thrust::device,
                                clusterid_of_points_full_size.begin(),
                                clusterid_of_points_full_size.end(),
                                data_points.begin());
        
        // sort the indexes of data_points: my_sequence_full_size
        thrust::stable_sort_by_key(thrust::device,
                                   clusterid_of_points_full_size2.begin(),
                                clusterid_of_points_full_size2.end(),
                                my_sequence_full_size.begin());
        
        
        // constant iterator
        thrust::fill(thrust::device,
                     singles_full_size.begin(),
                     singles_full_size.end(), 1);
        
        // get number of points in each cluser
        auto index_num_points = thrust::reduce_by_key(thrust::device,
                             clusterid_of_points_full_size.begin(),
                             clusterid_of_points_full_size.end(),
                             singles_full_size.begin(),
                             index_of_each_cluster.begin(),
                             num_points_each_cluster.begin());
        
        // get the sum of all points in each cluster
        auto get_sum = thrust::reduce_by_key(thrust::device,
                             clusterid_of_points_full_size.begin(),
                             clusterid_of_points_full_size.end(),
                             data_points.begin(),
                             index_of_each_cluster.begin(),
                             k_sum_centroids.begin(),
                              binary_pred,
                             p_add
                             );
  
        // centroids reordered based on indexes
        // this step can be avoided if index found above are guaranteed to be sorted
        thrust::stable_sort_by_key(thrust::device,
                                   index_of_each_cluster.begin(),
                                   index_of_each_cluster.end(),
                                   k_sum_centroids.begin());
        
        
        k_previous_centroids = k_centroids;
        
        // calculate centroid of each cluster
        thrust::transform(thrust::device,
                          k_sum_centroids.begin(),
                          k_sum_centroids.end(),
                          num_points_each_cluster.begin(),
                          k_centroids.begin(),
                          calcCentroid());
        
        
        
        
        // make the data_points reordering back to as it was before
        thrust::stable_sort_by_key(thrust::device,
                                   my_sequence_full_size.begin(),
                                   my_sequence_full_size.end(),
                                   data_points.begin());

        // check for convergence
        thrust::transform(thrust::device,
                         k_centroid_sequence.begin(),
                         k_centroid_sequence.end(),
                         k_centroid_check_convergence.begin(),
                         checkConvergence(thrust::raw_pointer_cast(&k_previous_centroids[0]),
                                          thrust::raw_pointer_cast(&k_centroids[0]),
                                          options.threshold,
                                          options.dims));
        
        int has_not_converged = thrust::reduce(k_centroid_check_convergence.begin(), k_centroid_check_convergence.end());
        

         if (has_not_converged == 0) {
             break;
         }
    }
    
    cudaEventRecord(cu_stop);     
    cudaEventSynchronize(cu_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
    printf("%d,%lf\n", (iter+1), milliseconds/(iter+1));

    thrust::copy(k_centroids.begin(),
                 k_centroids.end(),
                 host_k_centroids.begin());
    
    thrust::copy(clusterid_of_points.begin(),
                 clusterid_of_points.end(),
                 host_clusterid_of_points.begin());


    
    if (options.print_centroid) {
        for (int i = 0; i < options.num_cluster; i++) {
            printf("%d ", i);
            for (int j=0; j < options.dims; j++) {
                printf("%lf ", host_k_centroids[i * options.dims + j]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < num_vals; p++) {
            printf(" %d", host_clusterid_of_points[p]);
        }
    }
    
    return 0;
}
