#include <chrono>
#include "io.h"
#include "kmeans.h"

using namespace std;

static unsigned long int next_rand = 1;
static unsigned long kmeans_rmax = 32767;
void kmeans_srand(unsigned int seed) {
    next_rand = seed;
}
int kmeans_rand() {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand/65536) % (kmeans_rmax+1);
}


void init_centroids(struct options_t* args,
                    REAL*             dataset,
                    REAL*             centroids) {

    for (int k = 0; k < args->num_cluster; k++) {
        int index = kmeans_rand() % args->num_points;
        for (int d = 0; d < args->dims; d++) {
            centroids[k*args->dims+d] = dataset[index*args->dims+d];
        }
    }

}


void find_nearest_centroids(struct options_t* args,
                            REAL*             dataset,
                            REAL*             centroids,
                            int*              labels){

	for (int i = 0; i < args->num_points; i++) {
        REAL   nearest_distance = 1000000.0;
        int    nearest_centroid = 0;
        for (int k = 0; k < args->num_cluster; k++) {
            REAL accum_diff = 0.0;
            REAL dist;
    		for (int d = 0; d < args->dims; d++) {
                accum_diff += pow(dataset[i*args->dims+d] - centroids[k*args->dims+d], 2);
            }
            //dist = sqrt(accum_diff);
            dist = accum_diff;
            if(dist < nearest_distance){
                nearest_distance = dist;
                nearest_centroid = k;
            }
        }
        labels[i] = nearest_centroid;
    }
}

void average_labeled_centroids(struct options_t* args,
                               REAL*             dataset,
                               REAL*             centroids,
                               int*              labels,
                               int*              hit_counts,
                               REAL*             sum_centroids) {
/*
    for (int i = 0; i < args->num_points; i++) {
        hit_counts[labels[i]]++;
        for (int d = 0; d < args->dims; d++) {
            if(hit_counts[labels[i]] == 1){
                centroids[labels[i]*args->dims+d] = 0.0;
            }
            centroids[labels[i]*args->dims+d] += (dataset[i*args->dims+d] - centroids[labels[i]*args->dims+d]) / hit_counts[labels[i]];
        }
    }
*/
    for (int i = 0; i < args->num_points; i++) {
        hit_counts[labels[i]]++;
        for (int d = 0; d < args->dims; d++) {
            if(hit_counts[labels[i]] == 1){
                sum_centroids[labels[i]*args->dims+d] = 0.0;
            }
            sum_centroids[labels[i]*args->dims+d] += REAL(dataset[i*args->dims+d]);
        }
    }

    for (int i = 0; i < args->num_cluster; i++) {
        for (int d = 0; d < args->dims; d++) {
            centroids[i*args->dims+d] = REAL(sum_centroids[i*args->dims+d] / hit_counts[i]);
        }
    }

}

bool is_converged(struct options_t* args,
                  REAL*             centroids,
                  REAL*             old_centroids){

    for (int k = 0; k < args->num_cluster; k++) {
        for (int d = 0; d < args->dims; d++) {
            if( abs(centroids[k*args->dims+d]-old_centroids[k*args->dims+d]) > REAL(args->threshold)){
                return false;
            }
        }
    }
    return true;
}

void kmeans_seq(struct options_t* args,
                REAL**            dataset,
                int**             labels,
                REAL**            centroids, 
                float*            time_loops, 
                int*              iter_to_converge) {

    int* hit_counts;
    REAL* sum_centroids;
    hit_counts = (int*) malloc(args->num_cluster * sizeof(int));
	sum_centroids = (REAL*) malloc(args->num_cluster * sizeof(REAL) * args->dims);

    // initialize centroids randomly
    kmeans_srand(args->seed);
    init_centroids(args, *dataset, *centroids);

    // core algorithm
    int iterations = 0;
    bool done = false;
    REAL* old_centroids;
	old_centroids = (REAL*) malloc(args->num_cluster * sizeof(REAL) * args->dims);
    auto start = std::chrono::high_resolution_clock::now();
    while(!done) {

        memcpy(old_centroids, *centroids, args->num_cluster * sizeof(REAL) * args->dims);
        memset(hit_counts, 0, args->num_cluster * sizeof(int));
        iterations++;

        // labels is a mapping from each point in the dataset 
        // to the nearest (euclidean distance) centroid
        find_nearest_centroids(args, *dataset, *centroids, *labels);
        //cout << "iter: " << iterations << ", labels: "; for(int p=0; p<args->num_points; p++){cout<< (*labels)[p] << " "; } cout << endl;

        // the new centroids are the average 
        // of all the points that map to each 
        // centroid
        average_labeled_centroids(args, *dataset, *centroids, *labels, hit_counts, sum_centroids);
        done = iterations > args->max_num_iter || is_converged(args, *centroids, old_centroids);

    }
    //cout << "labels: "; for(int p=0; p<args->num_points; p++){cout<< (*labels)[p] << " "; } cout << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    *time_loops = diff.count()/1000.0;
    *iter_to_converge = iterations;

    return;

} 

