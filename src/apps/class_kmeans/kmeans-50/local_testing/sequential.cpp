#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <chrono>
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

double getDistanceSq(std::vector<double>& point1, std::vector<double>& point2, int dim) {
    double distance_sq = 0;
    for(int i=0; i < dim; i++) {
        distance_sq += (point1[i] - point2[i])*(point1[i] - point2[i]);
    }
    
    return distance_sq;
}

int getNearestCentroid(std::vector<double>& point, std::vector<std::vector<double>>& k_centroids, int k, int dim) {
    int nearest_index = 0;
    double smallest_dist =  getDistanceSq(point, k_centroids[nearest_index], dim);
    for(int i=1; i < k; i++) {
        double temp = getDistanceSq(point, k_centroids[i], dim);
        if (temp < smallest_dist) {
            nearest_index = i;
            smallest_dist = temp;
        }
    }
    
    return nearest_index;
}

void read_file(struct options_t& args,
               int*              n_vals,
               std::vector<std::vector<double>>&         input_vals,
               std::vector<std::vector<double>>& k_centroids,
               std::vector<std::vector<double>>& k_old_centroids,
               std::vector<int>&             clusterid_of_points) {

    // Open file
    std::ifstream in;
    in.open(args.in_file);
    // Get num vals
    in >> *n_vals;

    // Alloc input array
    input_vals = std::vector<std::vector<double> >(*n_vals , std::vector<double> (args.dims));


    // Read input vals
    for (int i = 0; i < *n_vals; ++i) {
        int temp_index; // unused for now
        in >> temp_index;
        for(int j=0; j < args.dims; j++) {
            in >> input_vals[i][j];
		}
    }

	// k centroids array init	
	k_centroids = std::vector<std::vector<double> >(args.num_cluster , std::vector<double> (args.dims));
	k_old_centroids = std::vector<std::vector<double> >(args.num_cluster , std::vector<double> (args.dims));
    
	clusterid_of_points=std::vector<int> (*n_vals);
    
    kmeans_srand(args.seed);	
    for (int i=0; i< args.num_cluster; i++) {
        int index = kmeans_rand() % (*n_vals);
        for(int j=0; j < args.dims; j++) {
            k_centroids[i][j] = input_vals[index][j];
        }
    }
}



int main(int argc, char** argv) {
    struct options_t options;
    get_opts(argc,
             argv,
             &options);
    int num_vals = 0;
    std::vector<std::vector<double>> data_points;
    std::vector<std::vector<double>> k_centroids;
    std::vector<std::vector<double>> k_old_centroids;
    std::vector<int> clusterid_of_points;
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
			std::cout << data_points[i][j] << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << "initial centroids:\n";

    // Print k_centroids
    for (int i = 0; i < options.num_cluster; i++) {
        for (int j=0; j < options.dims; j++) {
			std::cout << k_centroids[i][j] << " ";
		}
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
*/

    int *num_points_each_cluster = (int *)malloc(options.num_cluster * sizeof(int));
    
    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;
    for(; iter < options.max_num_iter; iter++) {
           
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
            
            // setting the number of points in each cluster
            num_points_each_cluster[centroid_index]++;
            for(int z=0; z < options.dims; z++) {
                k_centroids[centroid_index][z] += data_points[j][z];
            }
        }
        
        bool converged = true; 
        for(int j =0; j < options.num_cluster; j++) {
            if (num_points_each_cluster[j] != 0) {
                double l2_norm = 0.0;
                for(int z=0; z < options.dims; z++) {
                    double temp = k_centroids[j][z]/num_points_each_cluster[j];
                    l2_norm += (temp - k_old_centroids[j][z]) * (temp - k_old_centroids[j][z]);
                    k_centroids[j][z] = k_centroids[j][z]/num_points_each_cluster[j];
                }
                
                if (l2_norm > options.threshold * options.threshold) {
                    converged = false;
                }
            }
        }
        
        if (converged) {
            break;
        }
        
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    
    printf("%d,%lf\n", (iter+1), (double)diff.count()/(iter+1));
    
    if (options.print_centroid) {
        for (int i = 0; i < options.num_cluster; i++) {
            printf("%d ", i);
            for (int j=0; j < options.dims; j++) {
                printf("%lf ", k_centroids[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < num_vals; p++) {
            printf(" %d", clusterid_of_points[p]);
        }
    }
                                              
    free(num_points_each_cluster);
    return 0;
}
