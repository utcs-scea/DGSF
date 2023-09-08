#include <kmean_seq.h>
#include <math.h>

using namespace std;

kmean_seq::kmean_seq(struct kmean_t *km) {

    rmax = 32767;
    kmean = km;
    next = kmean->n_seed;
}

kmean_seq::~kmean_seq() {
    
}


int kmean_seq::rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (rmax+1);
}

void kmean_seq::assign_RandomCentroids() {
    
    for (int cl_idx = 0; cl_idx < kmean->n_clusters; cl_idx++) {
        
        
        int index = rand() % kmean->n_points; 
        //cout << "Centroid=" << cl_idx << ", Point=" << index+1 << endl;
        for (int col =0; col < kmean->n_dims; col++) {
            kmean->centroids[cl_idx][col] = kmean->input_pts[index][col];            
            //cout << "(" << kmean->centroids[cl_idx][col] << " , " << kmean->input_pts[index][col] << " )" << endl;
        }
        
    }
        
    
    
}


void kmean_seq::set_seed(unsigned int seed) {
    next = seed;
    cout << "kmean_seq: seed set to " << next;
}

void kmean_seq::findNearestCentroids() {
    
    float dist;      // Euclidian Distance
    float dist_min;  // Minimum Distance 
    float cl_min;
    float tmp;
    
    for (int row = 0; row < kmean->n_points ; row++) {
        
        for (int cl = 0; cl < kmean->n_clusters ; cl++) {
            

            dist = calc_EuclDist(kmean->input_pts[row], kmean->centroids[cl], kmean->n_dims);
   
            
            if (cl == 0) {
                dist_min = dist;
                cl_min = 0;
            } 
            
            if (dist_min > dist) {
                dist_min = dist;
                cl_min = cl;
            }
        }
        
        kmean->cluster_id[row] = cl_min;
        
    }
}

void kmean_seq::averageLabeledCentroids() {
    
    
    // Swap kmean->centroids and kmean->old_centroids
    float **s = kmean->centroids;
    kmean->centroids = kmean->old_centroids;
    kmean->old_centroids = s;
    
    // Zero out mean->centroids
    for (int cl = 0; cl < kmean->n_clusters; cl ++) {
        memset(kmean->centroids[cl], 0, kmean->n_dims * sizeof(float));
    }
    
    // Average all the points that map to each centroid
    int *num_pts_per_cluster = new int [kmean->n_clusters];
    memset(num_pts_per_cluster , 0, kmean->n_clusters * sizeof(int));
    
    for(int row = 0; row < kmean->n_points; row++) {
        
        for (int col = 0 ; col < kmean->n_dims; col++) {
            kmean->centroids[ kmean->cluster_id[row] ][col] += kmean->input_pts[row][col];
        }
        num_pts_per_cluster[ kmean->cluster_id[row] ]++; 
       
    }
  

    
    
    for (int cl = 0; cl < kmean->n_clusters; cl ++) {
        if ( num_pts_per_cluster[cl] != 0) {
            for (int col = 0; col < kmean->n_dims; col++) {
                kmean->centroids[cl][col] /= num_pts_per_cluster[cl];
            }
        }
    }
    
    delete [] num_pts_per_cluster;

}

float kmean_seq::calc_EuclDist(float *pt1, float *pt2, int n_elem) {
    
    float dist = 0;
    float tmp;

    for( int col = 0; col < n_elem; col++) {
        tmp= (pt1[col] - pt2[col]);
        dist += tmp*tmp;
    }
    dist = sqrt(dist);

    
    return dist;
}

// calulate the maximum Euclidian distance between old and new centroids
float kmean_seq::calc_Old2NewCentroidsDist() {
    
    float dist_max, dist;
    
    for (int cl = 0; cl < kmean->n_clusters; cl++) {
        
        dist = calc_EuclDist(kmean->centroids[cl] , kmean->old_centroids[cl] , kmean->n_dims);
        //cout << "dist= " << dist << " ";
        //for (int col=0; col < kmean->n_dims ; col ++) {
        //    cout <<"(" << kmean->centroids[cl][col] << "," << kmean->old_centroids[cl][col] << ") " ;
        //}
        //cout << endl;
        if (cl == 0) dist_max = dist;
                
        if ( dist_max < dist) dist_max = dist;
    
    }
    
    return dist_max;
    
}