#ifndef _KMEAN_KERNEL_H
#define _KMEAN_KERNEL_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <string.h>
#include <iostream>
#include <kmean.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 16*32

__global__ void dev_sumDistances(int Np, int Nc, int Nd, int Npc, int Npdc, 
                                 float *d_input_pts,
                                 float *d_centroids,
                                 float *d_sumdist);

__global__ void shared_sumDistances(int Np, int Nc, int Nd, int Npc, int Npdc, 
                                    float *d_input_pts,
                                    float *d_centroids,
                                    float *d_sumdist);

__global__ void dev_findMinDist(int Np, int Nc, 
                                float *d_sumdist, 
                                int *d_cluster_id);


__global__ void dev_addCentroids(int Nd, int Npd,
                                int *d_cluster_id, 
                                int *d_num_pts_p_cluster,
                                float *d_input_pts,
                                float *d_centroids);

__global__ void dev_addCentroids(int Nd,
                                int *d_cluster_id, 
                                int *d_num_pts_p_cluster,
                                float *d_input_pts,
                                float *d_centroids);

__global__ void dev_avgCentroids(int Nd, int Ncd,  
                                int *d_num_pts_p_cluster,
                                int *d_cluster_id,
                                float *d_centroids);



__global__ void dev_Old2NewCentroidsDist(int Nd, int Ncd, 
                                         float *d_centroids,
                                         float *d_old_centroids );


class kmean_kernel {
    
    private:
    
    unsigned long int next;
    unsigned long   rmax;
    struct kmean_t *kmean;
    int Nd;           // kmane->d_dims;
    int Np;           // kmean->n_points;
    int Nc;           // kmean->n_clusters;
    
    // Device variable
    float     *d_input_pts;
    int       *d_cluster_id;
    int       *d_num_pts_p_cluster;
    float     *d_centroids;
    float *d_old_centroids;
    float     *d_distances;
    float       *d_sumdist;
    float          *d_ptr1;
    float          *d_ptr2;
    
    // Host variables
    float               *h_sumdist;
    int              *h_cluster_id;
    float             *h_centroids;
    float         *h_old_centroids;
    int       *h_num_pts_p_cluster;
    float              *h_EuclDist;
    
    int rand(void);
    
    cudaEvent_t start, stop;
    float elapsedTime;    
  

    
    public:
    

    kmean_kernel(struct kmean_t *km);
    ~kmean_kernel(void);
    void assign_RandomCentroids(void);
    void copy_from_Host_to_Dev(void);
    void copy_from_Dev_to_Host(void);
    void tic(void);
    float toc(void);
    float processKMean(void);

};


#endif
