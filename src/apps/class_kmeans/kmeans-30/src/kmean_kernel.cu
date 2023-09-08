#include <kmean_kernel.h>

using namespace std;

#define SIZE sizeof(float)

kmean_kernel::kmean_kernel(struct kmean_t *km) {

    rmax = 32767;
    kmean = km;
    next = kmean->n_seed;
    Nd = kmean->n_dims;
    Np = kmean->n_points;
    Nc = kmean->n_clusters;
    
    // Allocate Device memories
    cudaMalloc(&d_input_pts,  (Np * Nd) *SIZE);   
    cudaMalloc(&d_cluster_id, Np * sizeof(int)); 
    cudaMalloc(&d_num_pts_p_cluster, Nc * sizeof(int));
    cudaMalloc(&d_centroids,  Nc * Nd*SIZE);  
    cudaMalloc(&d_old_centroids,  Nc * Nd*SIZE);   
    cudaMalloc(&d_distances, Np*Nc*Nd* SIZE);
    cudaMalloc(&d_sumdist, Np*Nc * SIZE);


    
    
    // Allocate Host memories  
    h_sumdist           = (float*) malloc( Np*Nc *SIZE);
    h_cluster_id        = (int*) malloc( Np *sizeof(int));
    h_centroids         = (float*) malloc(Nc*Nd*SIZE);
    h_old_centroids     = (float*) malloc(Nc*Nd*SIZE);
    h_num_pts_p_cluster = (int*) malloc(Nc*sizeof(int));
    h_EuclDist          = (float*) malloc(SIZE);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

   
}

kmean_kernel::~kmean_kernel() {
    
    cudaFree(d_input_pts);
    cudaFree(d_cluster_id);
    cudaFree(d_num_pts_p_cluster);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_distances);
    cudaFree(d_sumdist);
    cudaFree(d_num_pts_p_cluster);

    
    free(h_sumdist);
    free(h_cluster_id);
    free(h_centroids);
    free(h_old_centroids);
    free(h_num_pts_p_cluster);
    free(h_EuclDist);

    
}


int kmean_kernel::rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (rmax+1);
}

void kmean_kernel::assign_RandomCentroids() {
    
    
    for (int c = 0; c < Nc; c++) {
                
        int index = rand() % Np; 

        // copy into the device
        cudaMemcpy(&d_centroids[c * Nd] , &d_input_pts[index * Nd] , Nd * SIZE , cudaMemcpyDeviceToDevice);        
    }


}

void kmean_kernel::copy_from_Host_to_Dev() {


     for (int i = 0; i < Np; i++) {
         cudaMemcpy(&d_input_pts[i*Nd] , &kmean->input_pts[i][0], Nd * SIZE, cudaMemcpyHostToDevice);
     }
     


}

void kmean_kernel::copy_from_Dev_to_Host() {


     for (int i = 0; i < Nc; i++) {
         cudaMemcpy(&kmean->centroids[i][0] , &d_centroids[i*Nd], Nd * SIZE, cudaMemcpyDeviceToHost);
     }
     cudaMemcpy(&kmean->cluster_id[0] , &d_cluster_id[0], Nc * sizeof(int), cudaMemcpyDeviceToHost);

}

void kmean_kernel::tic() {
    cudaEventRecord(start, 0);
}

float kmean_kernel::toc() {

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time = %f (usec)\n", 1000*elapsedTime); 
    return elapsedTime;

}

float kmean_kernel::processKMean() {


  
  int Npc  = Np*Nc;
  int Npdc = Npc *Nd;
  int Npd  = Np*Nd;
  int Ncd  = Nc * Nd;
  int Ngrid = 1 << 8;

  cudaMemset(d_sumdist , 0.0, Npc * SIZE);

  if (kmean->n_rev == 3) {

     dev_sumDistances<<<(Npdc+Ngrid-1)/Ngrid , Ngrid>>>(Np, Nc, Nd, Npc, Npdc ,
                         &d_input_pts[0], &d_centroids[0], &d_sumdist[0]);
  } else {

    shared_sumDistances<<<(Npdc+Ngrid-1)/Ngrid , Ngrid>>>(Np, Nc, Nd, Npc, Npdc ,
                      &d_input_pts[0], &d_centroids[0], &d_sumdist[0]);
  }

  
  dev_findMinDist<<<(Np+Ngrid-1)/Ngrid , Ngrid>>>(Np, Nc, &d_sumdist[0], &d_cluster_id[0]);
   
  
 // Swap the pointers pointing to old and new centroid infomration
 float *tmp_ptr;
 tmp_ptr = d_centroids;
 d_centroids = d_old_centroids;
 d_old_centroids = tmp_ptr;
 

 cudaMemset(d_centroids, 0.0 , Nc*Nd *SIZE);
 cudaMemset(d_num_pts_p_cluster, 0, Nc * sizeof(int));

 dev_addCentroids<<<(Npd+Ngrid-1)/Ngrid , Ngrid>>>(Nd, Npd, &d_cluster_id[0], &d_num_pts_p_cluster[0], &d_input_pts[0], &d_centroids[0]);
 
 dev_avgCentroids<<<(Ncd+Ngrid-1)/Ngrid , Ngrid>>>(Nd, Ncd,  &d_num_pts_p_cluster[0], &d_cluster_id[0], &d_centroids[0]);

 dev_Old2NewCentroidsDist<<<(Ncd+Ngrid-1)/Ngrid , Ngrid>>>(Nd, Ncd, &d_centroids[0], &d_old_centroids[0] );
 
 cudaMemcpy(&h_centroids[0] , &d_old_centroids[0], Nc*Nd*SIZE, cudaMemcpyDeviceToHost);
 
 float err_max = 0;
 float dist;
 
 for (int i=0; i< Nc; i++) {
    dist = 0;
    for (int j=0; j<Nd; j++) {
        dist += h_centroids[i*Nd + j];
    }
    dist = sqrt(dist);
    
    if (i == 0) {
       err_max = dist;
    } else {
    
       if (err_max < dist) {
         err_max = dist;
       } 
    }
    
   } 
    
 return err_max;
 
}
/*-------------------------------------------------------------------------------------*/


/***** Calcualte  (x - y)^2 and add to the corresponding element ******/
__global__ void dev_sumDistances(int Np, int Nc, int Nd, int Npc, int Npdc, 
                                 float *d_input_pts,
                                 float *d_centroids,
                                 float *d_sumdist) {
                                     
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i_d, i_pc, i_c, i_p;
  float dxy;
  
  if (i < Npdc) {
  
     i_d  = i / Npc;
     i_pc = i % Npc; 
     i_c  = i_pc / Np;
     i_p  = i_pc % Np; 
     
    dxy = d_input_pts[i_p*Nd + i_d] - d_centroids[i_c*Nd + i_d];
    
   
    atomicAdd(&d_sumdist[i_p*Nc + i_c], dxy*dxy);
 
    __syncthreads();  
   }                                    
}

__global__ void shared_sumDistances(int Np, int Nc, int Nd, int Npc, int Npdc,
                                 float *d_input_pts,
                                 float *d_centroids,
                                 float *d_sumdist) {
                                     
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int it = threadIdx.x;
  
  int i_d, i_pc, i_c, i_p;
  float dxy;
  
  __shared__ float tmp[BLOCKSIZE];
  
  
  if (i < Npdc) {
  
     i_d  = i / Npc;
     i_pc = i % Npc; 
     i_c  = i_pc / Np;
     i_p  = i_pc % Np; 
    
    tmp[it] = d_centroids[i_c*Nd + i_d];
    __syncthreads();  
    
    dxy = d_input_pts[i_p*Nd + i_d] - tmp[it];
    
   
    atomicAdd(&d_sumdist[i_p*Nc + i_c], dxy*dxy);
 
    __syncthreads();  
   }                                    
}

/****Goes over all the points and find a cluster with minimum distant to that point ****/

__global__ void dev_findMinDist(int Np, int Nc, 
                                float *d_sumdist, 
                                int *d_cluster_id) {
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i < Np) {
  
     d_cluster_id[i] = 0; // Initially assume current cluster = 0 has the minimum distance to point i
     for(int c = 1; c < Nc; c++ ) {
     
        if ( d_sumdist[i*Nc + c] < d_sumdist[i*Nc + d_cluster_id[i]] ) {
            d_cluster_id[i] = c;
        }
     }
     __syncthreads();  
  }

}




/****** add all the points that belong to the same centroid  *******/
__global__ void dev_addCentroids(int Nd, int Npd,
                                int *d_cluster_id, 
                                int *d_num_pts_p_cluster,
                                float *d_input_pts,
                                float *d_centroids) {
                                
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i_d, i_p;
  
  if (i < Npd) {
      
     i_p = i / Nd;
     i_d = i % Nd;
  
  
     atomicAdd(&d_centroids[d_cluster_id[i_p] *Nd + i_d] , d_input_pts[i_p * Nd + i_d]); 
     if (i_d == 0)
       atomicAdd(&d_num_pts_p_cluster[d_cluster_id[i_p]], 1);
     
     __syncthreads();  
  }  
    
}

__global__ void dev_avgCentroids(int Nd, int Ncd,  
                                int *d_num_pts_p_cluster,
                                int *d_cluster_id,
                                float *d_centroids) {
                                
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i_c, i_d;
  
  if (i < Ncd) {
  
      i_c = i / Nd;
      i_d = i % Nd;
      
      if (d_num_pts_p_cluster[i_c] != 0)
          d_centroids[i_c*Nd + i_d] /= d_num_pts_p_cluster[i_c];

      __syncthreads();  
  }
}


__global__ void dev_Old2NewCentroidsDist(int Nd, int Ncd, 
                                         float *d_centroids,
                                         float *d_old_centroids ) {
                                         
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i_c, i_d;
  
  if (i < Ncd) {
  
      i_c = i / Nd;
      i_d = i % Nd;
      
      d_old_centroids[i_c*Nd + i_d] = d_centroids[i_c*Nd + i_d] -  d_old_centroids[i_c*Nd + i_d];
      d_old_centroids[i_c*Nd + i_d] *= d_old_centroids[i_c*Nd + i_d];
      __syncthreads();
   }

   
}
                                         
                                         
