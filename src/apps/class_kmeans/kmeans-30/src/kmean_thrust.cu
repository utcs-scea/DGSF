#include <kmean_thrust.h>

using namespace std;


kmean_thrust::kmean_thrust(struct kmean_t *km) {

    rmax = 32767;
    kmean = km;
    next = kmean->n_seed;
    Nd = kmean->n_dims;
    Np = kmean->n_points;
    Nc = kmean->n_clusters;
    
    // Allocate Device memories
    d_input_pts     = device_malloc<float>( Np * Nd);
    d_cluster_id    = device_malloc<int>(Np  );
    d_centroids     = device_malloc<float>(Nc * Nd );
    d_old_centroids = device_malloc<float>(Nc * Nd );   

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
}

kmean_thrust::~kmean_thrust() {
    
    device_free(d_input_pts);
    device_free(d_cluster_id);
    device_free(d_centroids);
    device_free(d_old_centroids);
    
}


int kmean_thrust::rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (rmax+1);
}

void kmean_thrust::assign_RandomCentroids() {
    
    for (int c = 0; c < Nc; c++) {
                
        int index = rand() % Np; 

        // copy inside the device
        thrust::copy(d_input_pts + index * Nd ,  d_input_pts + (index+1) * Nd , d_centroids + c * Nd );
                
    }
        
}


void kmean_thrust::findNearestCentroids() {
    

    float  dist;      // Euclidian Distance
    float  dist_min;  // Minimum Distance 
    int cl_min;


    for (int row = 0; row < Np ; row++) {
        
        for (int cl = 0; cl < Nc ; cl++) {
            

            dist =  calc_EuclDist(d_input_pts+row*Nd, d_centroids + cl*Nd, Nd);
           
           
            if (cl == 0) {
                dist_min = dist;
                cl_min = 0;
            } 
            
            if (dist_min > dist) {
                dist_min = dist;
                cl_min = cl;
            }
        }
        
        d_cluster_id[row] = cl_min;
        
    } 
 
}

void kmean_thrust::averageLabeledCentroids() {
    

    // Swap kmean->centroids and kmean->old_centroids
     device_ptr<float> s = d_centroids;
     d_centroids = d_old_centroids;
     d_old_centroids = s;
 

    
    thrust::fill(d_centroids , d_centroids + Nc * Nd, 0.0);
  
    // Average all the points that map to each centroid
    device_ptr<int> num_pts_per_cluster = device_malloc<int>(Nc); 
    thrust::fill( num_pts_per_cluster, num_pts_per_cluster + Nc, 0);
    
  
    for(int row = 0; row < Np; row++) {


        thrust::transform(d_centroids + (d_cluster_id[row]*Nd), 
                          d_centroids + ((d_cluster_id[row]+1)*Nd) , 
                          d_input_pts + (row*Nd), 
                          d_centroids + (d_cluster_id[row]*Nd), thrust::plus<float>());
         num_pts_per_cluster[d_cluster_id[row]]++; 

    }
    for (int cl = 0; cl < Nc; cl ++) {
        if ( num_pts_per_cluster[cl] != 0) {
        
           thrust::transform(d_centroids + cl*Nd , 
                              d_centroids + (cl+1) * Nd , 
                              thrust::make_constant_iterator(num_pts_per_cluster[cl]), 
                              d_centroids + cl*Nd , 
                              thrust::divides<float>());

       }

   }    

   device_free(num_pts_per_cluster);

}

float kmean_thrust::calc_EuclDist(device_ptr<float> pt1, device_ptr<float> pt2, int n_elem) {
    
    device_vector<float> tmp(n_elem); 
    device_vector<float> result(1);
    
 
    thrust::transform(pt1, pt1 + n_elem, pt2, tmp.begin(), thrust::minus<float>());
    float res = thrust::inner_product( tmp.begin(), tmp.end(), tmp.begin() , 0.0 ); 

    return sqrt(res);
    
}


float kmean_thrust::calc_Old2NewCentroidsDist() {
    
    float dist_max, dist;
    
    for (int cl = 0; cl < Nc; cl++) {
        
        dist = calc_EuclDist(d_centroids + cl*Nd , d_old_centroids + cl*Nd , Nd);

        if (cl == 0) dist_max = dist;
                
        if ( dist_max < dist ) dist_max = dist;
    
    }
    
    return dist_max;
    
}


void kmean_thrust::copy_from_Host_to_Dev() {


    for (int i = 0; i < Np; i++) {
      thrust::copy( &kmean->input_pts[i][0], &kmean->input_pts[i][0] + Nd , d_input_pts + i*Nd);
    }
}

void kmean_thrust::copy_from_Dev_to_Host() {


    for (int i = 0; i < Nc; i++) {
      thrust::copy(d_centroids + i*Nd , d_centroids + (i+1)*Nd, &kmean->centroids[i][0]);
    }
    thrust::copy(d_cluster_id , d_cluster_id + Np, &kmean->cluster_id[0]);


}

void kmean_thrust::tic() {
    cudaEventRecord(start, 0);
}

float kmean_thrust::toc() {

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Main loop elapsed time = %f (usec)\n", 1000*elapsedTime); 
    return elapsedTime;

}


