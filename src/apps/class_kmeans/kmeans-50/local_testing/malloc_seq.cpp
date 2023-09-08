#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
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

double getDistanceSq(double* point1, double* point2, int dim) {
    double distance_sq = 0;
    for(int i=0; i < dim; i++) {
        distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
        //std::cout << point1[i] << " " << point2[i] << "\n";
    }
                                             
    //std::cout << std::endl;
    
    return distance_sq;
}

int getNearestCentroid(double* point, double** k_centroids, int k, int dim) {
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

void read_file(struct options_t& args,
               int*              n_vals,
               double***         input_vals,
               double*** k_centroids,
               double*** k_old_centroids,
               int**             clusterid_of_points) {

    // Open file
    std::ifstream in;
    in.open(args.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array
    *input_vals = (double**)malloc((*n_vals) * sizeof(double*));
    (*input_vals)[0] = (double*)malloc((*n_vals)*(args.dims)  * sizeof(double));
    for(int i =1; i < *n_vals; i++) {
        (*input_vals)[i] = (*input_vals)[i-1] + args.dims;
    }

    // Read input vals
    for (int i = 0; i < *n_vals; ++i) {
        int temp_index; // unused for now
        in >> temp_index;
        for(int j=0; j < args.dims; j++) {
            in >> (*input_vals)[i][j];
		}
    }


	// k centroids array init
    *k_centroids = (double**)malloc((args.num_cluster) * sizeof(double*));
    (*k_centroids)[0] = (double*)malloc((args.num_cluster)*(args.dims)  * sizeof(double));
    for(int i =1; i < args.num_cluster; i++) {
        (*k_centroids)[i] = (*k_centroids)[i-1] + args.dims;
    }
    

    // k old centroids array init
    *k_old_centroids = (double**)malloc((args.num_cluster) * sizeof(double*));
    (*k_old_centroids)[0] = (double*)malloc((args.num_cluster)*(args.dims)  * sizeof(double));
    for(int i =1; i < args.num_cluster; i++) {
        (*k_old_centroids)[i] = (*k_old_centroids)[i-1] + args.dims;
    }
    
    
    
    
	*clusterid_of_points=(int*)malloc(*n_vals * sizeof(int));
    
    kmeans_srand(args.seed);	
    for (int i=0; i< args.num_cluster; i++) {
        int index = kmeans_rand() % (*n_vals);
        for(int j=0; j < args.dims; j++) {
            (*k_centroids)[i][j] = (*input_vals)[index][j];
        }
    }
}



int main(int argc, char** argv) {
    struct options_t options;
    get_opts(argc,
             argv,
             &options);
    int num_vals = 0;
    double** data_points;
    double** k_centroids;
    double** k_old_centroids;
    int* clusterid_of_points;
    read_file(options,
              &num_vals,
              &data_points,
              &k_centroids,
              &k_old_centroids,
              &clusterid_of_points);

	std::cout.precision(10);
/*
    // Print input vals
    for (int i = 0; i < num_vals; ++i) {
        for (int j=0; j < options.dims; j++) {
			std::cout << data_points[i][j] << " ";
		}
        std::cout << std::endl;
    }
*/    
    std::cout << "initial centroids:\n";

    // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
			std::cout << k_centroids[i][j] << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << std::endl;

    int *num_points_each_cluster = (int *)malloc(options.num_cluster * sizeof(int));
    //int num_points_each_cluster[5000];
    
    //std::cout << data_points << " " << k_centroids << " " << k_old_centroids << " " << clusterid_of_points 
    //    << " " << num_points_each_cluster << std::endl;
    for(int i=0; i < options.max_num_iter; i++) {
           
        // copy centroid as old centroid
        for(int j=0; j < options.num_cluster; j++) {
            for (int z=0; z < options.dims; z++) {
                k_old_centroids[j][z] = k_centroids[j][z];
                k_centroids[j][z] = 0;
            }
        }
   
        std::memset(num_points_each_cluster, 0, sizeof(int) * options.num_cluster);
        
        
            // Print k_centroids
    for (int j = 0; j < options.num_cluster; j++) {
        for (int z=0; z < options.dims; z++) {
			std::cout << k_centroids[j][z] << " ";
		}
        std::cout << std::endl;
    }
        
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
        
    }
    
    
    
    // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
			std::cout << k_centroids[i][j] << " ";
		}
        std::cout << std::endl;
    }

    free(data_points[0]);
    free(data_points);
    free(k_centroids[0]);
    free(k_centroids);
    free(k_old_centroids[0]);
    free(k_old_centroids);
    free(clusterid_of_points);
    free(num_points_each_cluster);
	
    return 0;
}
