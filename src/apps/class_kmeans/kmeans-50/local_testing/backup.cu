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
               thrust::device_vector<double>& data_points,
               thrust::device_vector<double>& k_centroids,
               thrust::device_vector<double>& k_old_centroids,
               thrust::device_vector<int>&    clusterid_of_points) {

    
    std::ifstream in;
    in.open(options.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array  
    data_points = thrust::device_vector<double>(*n_vals * options.dims);
    
    
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
    
   
    k_centroids = thrust::device_vector<double>(options.num_cluster * options.dims);
    //extra size just for now
    k_old_centroids = thrust::device_vector<double>(options.num_cluster * options.dims);

    
 
	clusterid_of_points=thrust::device_vector<int>(*n_vals);
    
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
    thrust::device_vector<double> data_points;
    thrust::device_vector<double> k_centroids;
    thrust::device_vector<double> k_old_centroids;
    thrust::device_vector<int> clusterid_of_points;
    
   
    
    read_file(options,
              &num_vals,
              data_points,
              k_centroids,
              k_old_centroids,
              clusterid_of_points);

    
    
    
	std::cout.precision(10);  
    std::cout << "initial centroids:\n";
    
    

    // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
            double temp = k_centroids[i * options.dims + j];
			std::cout << temp << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << std::endl;

    
        
        
    
    // extra size ?
    thrust::device_vector<int> num_points_each_cluster = thrust::device_vector<int>(options.num_cluster * options.dims);
    
    // extra values for some testing ?
    thrust::device_vector<int> index_of_each_cluster = thrust::device_vector<int>(options.num_cluster * options.dims);
    thrust::equal_to<int> binary_pred;
     
    
    point_add p_add(options.dims);
    
    thrust::device_vector<int>  my_sequence(num_vals);
    //thrust::device_vector<int>  temp_sequence(num_vals);
    thrust::sequence(my_sequence.begin(), my_sequence.end(), 0);
    
    
    thrust::device_vector<int>  my_sequence_full_size(num_vals * options.dims);
    thrust::device_vector<int>  temp_my_sequence_full_size(num_vals * options.dims);
    
    thrust::sequence(my_sequence_full_size.begin(), my_sequence_full_size.end(), 0);
    
    thrust::device_vector<int> singles_full_size(num_vals * options.dims, 1);
    thrust::device_vector<int> clusterid_of_points_full_size(num_vals * options.dims);
    thrust::device_vector<int> clusterid_of_points_full_size2(num_vals * options.dims);    

    auto start = std::chrono::high_resolution_clock::now();
        auto start2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i < options.max_num_iter; i++) {
        
        cudaEvent_t cu_start, cu_stop;
cudaEventCreate(&cu_start);
cudaEventCreate(&cu_stop);

        thrust::transform(thrust::device,
            my_sequence.begin(),
                          my_sequence.end(),
                          clusterid_of_points.begin(),
                          getCentroidIndex(thrust::raw_pointer_cast(&k_centroids[0]),
                                 thrust::raw_pointer_cast(&data_points[0]),
                                 options.num_cluster,
                                 options.dims));
        

        

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    printf("%lf\n", (double)diff.count());

            
        
/*      
        // typedef these iterators for shorthand
typedef thrust::device_vector<int>::iterator   IntIterator;

// typedef a tuple of these iterators
typedef thrust::tuple<IntIterator, IntIterator> IteratorTuple;
// typedef the zip_iterator of this tuple
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
// finally, create the zip_iterator
ZipIterator iter(thrust::make_tuple(clusterid_of_points.begin(), my_sequence.begin()));
*/
        
   //     thrust::sort_by_key(clusterid_of_points.begin(),
   //                         clusterid_of_points.end(),
   //                         my_sequence.begin());
    
                cudaEventRecord(cu_start);
        
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
        
        //for(int j=0; j < options.dims * num_vals; j++) {
            //clusterid_of_points_full_size[j] = clusterid_of_points[j /options.dims]*options.dims +  (j % options.dims);
            //clusterid_of_points_full_size2[j] = clusterid_of_points[j /options.dims]*options.dims +  (j % options.dims);
            //std::cout << clusterid_of_points_full_size[j] << " ";
        //}
                          
                          

        
           
                        cudaEventRecord(cu_stop);     
                cudaEventSynchronize(cu_stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
        
            printf("%lf  this is the cost of copy\n", milliseconds);
        
        
        
        
                        cudaEventRecord(cu_start);
        
    start = std::chrono::high_resolution_clock::now();

        thrust::stable_sort_by_key(thrust::device,
                                clusterid_of_points_full_size.begin(),
                                clusterid_of_points_full_size.end(),
                                data_points.begin());
        
        thrust::stable_sort_by_key(thrust::device,
                                   clusterid_of_points_full_size2.begin(),
                                clusterid_of_points_full_size2.end(),
                                my_sequence_full_size.begin());
        
        
        
        thrust::fill(thrust::device,
                     singles_full_size.begin(),
                     singles_full_size.end(), 1);
        
          cudaEventRecord(cu_stop);
        cudaEventSynchronize(cu_stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
        
            printf("%lf  this is the cost of 2 stable sort\n", milliseconds);
        
/*        
        for (int j =0; j < options.num_cluster; j++) {
            std::cout << num_points_each_cluster[j] << " ";
        }
        
        std::cout << std::endl;
        
        
        for (int j =0; j < options.num_cluster; j++) {
            std::cout << index_of_each_cluster[j] << " ";
        }
        
        std::cout << std::endl;
*/        
        
                                      cudaEventRecord(cu_start);        
        auto ab = thrust::reduce_by_key(thrust::device,
                             clusterid_of_points_full_size.begin(),
                             clusterid_of_points_full_size.end(),
                             singles_full_size.begin(),
                             index_of_each_cluster.begin(),
                             num_points_each_cluster.begin());
        
        
                      
 
        
  
        
        auto get_sum = thrust::reduce_by_key(thrust::device,
                             clusterid_of_points_full_size.begin(),
                             clusterid_of_points_full_size.end(),
                             data_points.begin(),
                             index_of_each_cluster.begin(),
                             k_old_centroids.begin(),
                              binary_pred,
                             p_add
                             );
        
        
                                cudaEventRecord(cu_stop);   
        cudaEventSynchronize(cu_stop);
 milliseconds = 0;
cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
        
            printf("%lf  this is the cost of 2 reduce_by_key\n", milliseconds);
        
/*
        std::cout << "\n";
                // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
            double temp = k_old_centroids[i * options.dims + j];
			std::cout << temp << " ";
		}
        std::cout << std::endl;
    }
*/    
        
        
        
            cudaEventSynchronize(cu_start);    
        
        thrust::stable_sort_by_key(thrust::device,
                                   index_of_each_cluster.begin(),
                                   index_of_each_cluster.end(),
                                   k_old_centroids.begin());
        
        
        thrust::transform(thrust::device,
                          k_old_centroids.begin(),
                          k_old_centroids.end(),
                          num_points_each_cluster.begin(),
                          k_centroids.begin(),
                          calcCentroid());
        
        thrust::stable_sort_by_key(thrust::device,
                                   my_sequence_full_size.begin(),
                                   my_sequence_full_size.end(),
                                   data_points.begin());
        
        
        
                                        cudaEventRecord(cu_stop);
        cudaEventSynchronize(cu_stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
        
            printf("%lf  this is the cost of 2 sort and 1 transform \n", milliseconds);
            end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
            printf("%lf\n", (double)diff.count());
        

        
        
    }
    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto diff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2);
    printf("%lf\n  wow why so", (double)diff2.count());
    
    
        // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
            double temp = k_centroids[i * options.dims + j];
			std::cout << temp << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    
    
    return 0;
}
