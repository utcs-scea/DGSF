#ifndef _KMEAN_THRUST_H
#define _KMEAN_THRUST_H

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
#include <thrust/functional.h>
#include <string.h>
#include <iostream>
#include <kmean.h>
#include <math.h>
#include <vector>

using namespace thrust;

class kmean_thrust {
    
    private:
    
    unsigned long int next;
    unsigned long   rmax;
    struct kmean_t *kmean;
    int Nd;           // kmane->d_dims;
    int Np;           // kmean->n_points;
    int Nc;           // kmean->n_clusters;
    
    device_ptr<float>     d_input_pts;
    device_ptr<int>       d_cluster_id;
    device_ptr<float>     d_centroids;
    device_ptr<float> d_old_centroids;

    
    int rand(void);
    
    cudaEvent_t start, stop;
    float elapsedTime;    
  

    public:
    

    kmean_thrust(struct kmean_t *km);
    ~kmean_thrust(void);
    void assign_RandomCentroids(void);
    void findNearestCentroids(void);
    void averageLabeledCentroids(void);
    float calc_EuclDist(device_ptr<float> pt1, device_ptr<float> pt2, int n_elem);
    float calc_Old2NewCentroidsDist(void);
    void copy_from_Host_to_Dev(void);
    void copy_from_Dev_to_Host(void);
    void tic(void);
    float toc(void);
    void test(void);


};
#endif

