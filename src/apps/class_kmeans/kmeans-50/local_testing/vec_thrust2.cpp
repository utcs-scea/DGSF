#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

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


struct point_add : public thrust::binary_function<double*,double*,double*>
{

  using Base = thrust::binary_function<double*,double*,double*>;
  int dims;
  double* temp_for_double_add;
  
  point_add(int dim, double* temp_double_add) :
  Base(),
  dims(dim),
  temp_for_double_add(temp_double_add) {}
  
  __host__ __device__
  double* operator()(const double* x,
                                        const double* y) {
    //double* c = x;
    for (int i = 0; i < dims; i++) {
    
        temp_for_double_add[i] = x[i] + y[i];
       if (x[i] + y[i] > 10000000) {
    //      printf("awesome");
       }
    }
    
    return temp_for_double_add;
  }
};

 struct saxpy2 {
     double** k_centroids;
     int num_cluster;
     int dim;
     saxpy2(double** centroids,
           int n_cluster,
           int dim):
     k_centroids(centroids),
     num_cluster(n_cluster),
     dim(dim) {}
     
     __host__ __device__
     double getDistanceSq(const double* point1, const double* point2) const {
        double distance_sq = 0;
        for(int i=0; i < dim; i++) {
            distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
            //std::cout << point1[i] << " " << point2[i] << "\n";
        }
                                             
        //std::cout << std::endl;
    
        return distance_sq;
     }
     
     __host__ __device__
     int operator()(const double* point) const {
        int nearest_index = 0;
        double smallest_dist =  getDistanceSq(point, k_centroids[nearest_index]);
         //const double*& point2 =  k_centroids + nearest_index;
         
        for(int i=1; i < num_cluster; i++) {
            double temp = getDistanceSq(point, k_centroids[i]);
            //std::cout << i << " temp:" << temp << "smallest_dist:" << smallest_dist <<std::endl;
            if (temp < smallest_dist) {
                nearest_index = i;
                smallest_dist = temp;
            }
        }
    
        return nearest_index;
     }
 };



/*
double getDistanceSq(thrust::device_vector<double>& point1, thrust::device_vector<double>& point2, int dim) {
    double distance_sq = 0;
    for(int i=0; i < dim; i++) {
        distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
        //std::cout << point1[i] << " " << point2[i] << "\n";
    }
                                             
    //std::cout << std::endl;
    
    return distance_sq;
}

int getNearestCentroid(thrust::device_vector<double>& point, thrust::device_vector<thrust::device_vector<double>>& k_centroids, int k, int dim) {
    int nearest_index = 0;
    double smallest_dist =  getDistanceSq(point, k_centroids[nearest_index], dim);
    for(int i=1; i < k; i++) {
        double temp = getDistanceSq(point, k_centroids[i], dim);
        //std::cout << i << " temp:" << temp << "smallest_dist:" << smallest_dist <<std::endl;
        if (temp < smallest_dist) {
            nearest_index = i;
            smallest_dist = temp;
        }
    }
    
    return nearest_index;
}


*/
void read_file(struct options_t& options,
               int*              n_vals,
               thrust::device_vector<double*>& data_points,
               thrust::device_vector<double*>& k_centroids,
               thrust::device_vector<double*>& k_old_centroids,
               thrust::device_vector<int>&             clusterid_of_points) {

    
    std::ifstream in;
    in.open(options.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array
    
    thrust::device_ptr<double> all_points = thrust::device_malloc<double>(*n_vals * options.dims);
    data_points = thrust::device_vector<double*>(*n_vals);
    data_points[0] = all_points;
    for(int i =1; i < *n_vals; i++) {
        data_points[i] = all_points + options.dims * i;
    }
    


    
    // Read input vals
    for (int i = 0; i < *n_vals * options.dims; ++i) {
        if (i % options.dims == 0) {
            // unused index
            double temp;
            in >> temp;  
        }
        double temp;
        in >> temp;
        all_points[i] = temp;
    }
    

	// k centroids array init
    
    
    double* all_centroid_points = thrust::device_malloc<double>(options.num_cluster * options.dims);
    k_centroids = thrust::device_vector<double*>(options.num_cluster);
    k_centroids[0] = all_centroid_points;
    for(int i =1; i < options.num_cluster; i++) {
        k_centroids[i] = all_centroid_points + options.dims * i;
    }

    
    // using extra size just for some testing
    double* all_sum_centroid_points = thrust::device_malloc<double>(*n_vals * options.dims);
    k_old_centroids = thrust::device_vector<double*>(*n_vals);
    k_old_centroids[0] = all_sum_centroid_points;
    for(int i =1; i < *n_vals; i++) {
        k_old_centroids[i] = all_sum_centroid_points + options.dims * i;
    }
    
 
	clusterid_of_points=thrust::device_vector<int>(*n_vals);
    
    kmeans_srand(options.seed);	
    for (int i=0; i< options.num_cluster; i++) {
        int index = kmeans_rand() % (*n_vals);
        for(int j=0; j < options.dims; j++) {
            all_centroid_points[i * options.dims + j] = all_points[index*options.dims + j];
        }
    }
}



int main(int argc, char** argv) {
    struct options_t options;
    get_opts(argc,
             argv,
             &options);
    int num_vals = 0;
    thrust::device_vector<double*> data_points;
    thrust::device_vector<double*> k_centroids;
    thrust::device_vector<double*> k_old_centroids;
    thrust::device_vector<int> clusterid_of_points;
    
    
    /*
 
     std::ifstream in;
    in.open(options.in_file);
    // Get num vals
    in >> num_vals;

    // Alloc input array
    
    double* all_points = thrust::device_malloc<double>(num_vals * options.dims);
    data_points = thrust::device_vector<double*>(num_vals);
    data_points[0] = all_points;
    for(int i =1; i < num_vals; i++) {
        data_points[i] = all_points + options.dims * i;
    }
    

    int temp_index; // unused for now
    in >> temp_index;
    
    // Read input vals
    for (int i = 0; i < num_vals * options.dims; ++i) {
        double temp;
        in >> temp;
        all_points[i] = temp;
    }
    
     */

    
    
    
    
    
    
    read_file(options,
              &num_vals,
              data_points,
              k_centroids,
              k_old_centroids,
              clusterid_of_points);

    
    
    
	std::cout.precision(10);
/*
    // Print input vals
    for (int i = 0; i < num_vals; ++i) {
        for (int j=0; j < options.dims; j++) {
            double* temp = data_points[i];
			std::cout << a[j] << " ";
		}
        std::cout << std::endl;
    }
*/    
    std::cout << "initial centroids:\n";
    
    

    // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
            double* temp = k_centroids[i];
			std::cout << temp[j] << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << std::endl;

    
        
        
    
    thrust::device_vector<int> num_points_each_cluster = thrust::device_vector<int>(options.num_cluster);
    
    // extra values for some testing
    thrust::device_vector<int> index_of_each_cluster = thrust::device_vector<int>(num_vals);
    thrust::equal_to<int> binary_pred;
    
    double* temp_for_double_add = thrust::device_malloc<double>(options.dims);
    point_add p_add(options.dims, temp_for_double_add);
    

    for(int i=0; i < options.max_num_iter; i++) {
        
        thrust::transform(data_points.begin(),
                          data_points.end(),
                          clusterid_of_points.begin(),
                          saxpy2((thrust::raw_pointer_cast(k_centroids.data())),
                                 options.num_cluster,
                                 options.dims));
        
        thrust::reduce_by_key(clusterid_of_points.begin(),
                             clusterid_of_points.end(),
                             data_points.begin(),
                              index_of_each_cluster.begin(),
                             k_old_centroids.begin(),
                              binary_pred,
                              p_add
                             );
        
        /*
        for (int j =0; j < num_vals; j++) {
            std::cout << clusterid_of_points[j] << " ";
        }
        
        
        // copy centroid as old centroid
        for(int j=0; j < options.num_cluster; j++) {
            for (int z=0; z < options.dims; z++) {
                k_old_centroids[j][z] = k_centroids[j][z];
                k_centroids[j][z] = 0;
            }
        }
   
        std::memset(num_points_each_cluster, 0, sizeof(int) * options.num_cluster);
        
        for(int j=0; j < num_vals; j++) {
            int centroid_index = getNearestCentroid(data_points[j],
                                                    k_old_centroids,
                                                    options.num_cluster,
                                                    options.dims);
            clusterid_of_points[j] = centroid_index;
            //std::cout << "cluster " << j << "is " << centroid_index << std::endl;
            
            // setting the number of points in each cluster
            num_points_each_cluster[centroid_index]++;
            for(int z=0; z < options.dims; z++) {
                k_centroids[centroid_index][z] += data_points[j][z];
            }
        }
        
        
        for(int j =0; j < options.num_cluster; j++) {
            if (num_points_each_cluster[j] != 0) {
                for(int z=0; z < options.dims; z++) {
                    k_centroids[j][z] = k_centroids[j][z]/num_points_each_cluster[j];
                }
            }
        }
        */
        
    }
    
    
    
    // Print k_centroids

	
    
    
    return 0;
}
