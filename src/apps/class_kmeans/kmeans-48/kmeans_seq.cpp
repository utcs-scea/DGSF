#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <chrono>


static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

inline double get_distance(double *point, double *centroid, int dims) {
    double sum = 0.0;
    for (int i=0; i<dims; i++) {
        sum += (point[i]-centroid[i])*(point[i]-centroid[i]);
    }
    return sum;
}

inline double convergence_score(double **old_centroids, double **new_centroids, int num_clusters, int dims) {
    double score = 0.0;
    for (int i=0; i<num_clusters; ++i) {
        double tmp_score = 0.0;
        for (int j=0; j<dims; ++j) {
            tmp_score += (old_centroids[i][j] - new_centroids[i][j])
                *(old_centroids[i][j] - new_centroids[i][j]);
        }
        score += tmp_score;
    }
    return score;
}

int kmeans_seq(double **points, int num_inputs, int dims, double** centroids, 
               double **old_centroids, int num_clusters, 
               double threshold, int max_num_iters, int *labels) {
    int curr_iter = 0;
    int num_points_assigned[num_clusters];
    int should_break = 0;
    while (curr_iter < max_num_iters) {

        
        // first copy centroids to old centroids
        memcpy(old_centroids[0], centroids[0], num_clusters*dims*sizeof(double));
//         for (int i=0; i<num_clusters; i++) {
//             memcpy(old_centroids[i], centroids[i], dims*sizeof(double));
//         }
        for (int i=0; i<num_inputs; i++) {
            double min_dist = 0.0;
            int min_centroid = -1;
            for (int j=0; j<num_clusters; j++) {
                double dist = get_distance(points[i], centroids[j], dims);
                if (min_centroid <0 || min_dist > dist) {
                    min_centroid = j;
                    min_dist = dist;
                }
            }
            labels[i] = min_centroid;
        }
        for (int i=0; i<num_clusters; i++) {
            num_points_assigned[i] = 0;
            for (int j=0; j<dims; j++) {
                centroids[i][j]=0.0;
            }
        }
        for (int i=0; i<num_inputs; i++) {
           for (int j=0; j<dims; j++) {
               centroids[labels[i]][j] += points[i][j];
           }
           num_points_assigned[labels[i]]+=1; 
        }
        for (int i=0; i<num_clusters; i++) {
            for (int j=0; j<dims; j++) {
                centroids[i][j] = centroids[i][j]/((double)num_points_assigned[i]);
            }
        }
        curr_iter += 1;
        if (convergence_score(old_centroids, centroids, num_clusters, dims) < threshold) {
            should_break+=1;
        } else {
            should_break = 0;
        }
        if (should_break > 3) {
            break;
        }
        // check fo convergence here and break
    }
    return curr_iter;
    
}


int main(int argc, char* argv[]) {
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

    // read the file and store the input
    FILE *fp = fopen(fileName, "r");
    int num_inputs;
    c = fscanf(fp, "%d", &num_inputs);

    double **points = (double**)malloc(num_inputs*sizeof(double*));
    double *tmp;
    tmp = (double*)malloc(num_inputs*dims*(sizeof(double)));
    for (int i=0; i<num_inputs; i++) {
        points[i] = tmp + (i*dims);
//         points[i] = (double*)malloc(dims*(sizeof(double)));
    }
    int point_num;
    for (int i=0; i<num_inputs; i++) {
        c = fscanf(fp, "%d", &point_num);
        for (int j=0; j<dims; j++) {
            c = fscanf(fp, "%lf", &(points[i][j]));
        }
    }
    // declare the centroids and use the rands to initialize the centroids
    double **centroids = (double**)malloc(num_clusters*sizeof(double*));
    double **old_centroids = (double**)malloc(num_clusters*sizeof(double*));
    tmp = (double*)malloc(num_clusters*dims*(sizeof(double)));
    double *tmp2 = (double*)malloc(num_clusters * dims * sizeof(double));
    for (int i=0; i<num_clusters; i++) {
        centroids[i] = tmp + (i*dims);
        old_centroids[i] = tmp2 + (i*dims);
//         centroids[i] = (double*)malloc(dims*(sizeof(double)));
//         old_centroids[i] = (double*) malloc(dims*sizeof(double));
    }
    kmeans_srand(seed); // cmd_seed is a cmdline arg
    for (int i=0; i<num_clusters; i++){
        int index = kmeans_rand() % num_inputs;
        // you should use the proper implementation of the following
        // code according to your data structure
        memcpy(centroids[i], points[index], dims*sizeof(double));
//         for (int j=0; j<dims; j++) {
//             centroids[i][j] = points[index][j];
//         }
    }
    // call the k means function to compute the kmeans
    int *labels;
    labels = (int*)malloc(num_inputs*sizeof(int));
    auto start = std::chrono::high_resolution_clock::now();
    int total_iters = kmeans_seq(points, num_inputs, dims, centroids, old_centroids, 
                                 num_clusters, threshold, max_num_iters, labels);
    auto end = std::chrono::high_resolution_clock::now();
    double time = ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count())/(double)total_iters;

    printf("%d,%lf\n", total_iters, time/1000.0);
    // print the output
    if (outputCentroids) {
        for (int i=0; i<num_clusters; i++) {
            printf("%d ", i);
            for (int j=0; j<dims; j++) {
                printf("%lf ", centroids[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < num_inputs; p++)
            printf(" %d", labels[p]);
    }
    
    // free the memory
    // Well don't free memory to save e2e execution time
//     for (int i=0; i<num_inputs; i++) {
//         free(points[i]);
//     }
//     free(points);
//     for (int i=0; i<num_clusters; i++) {
//         free(centroids[i]);
//         free(old_centroids[i]);
//     }
//     free(centroids);
//     free(old_centroids);
//     free(labels);
}