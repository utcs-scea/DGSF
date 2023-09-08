#ifndef _KMEAN_SEQ_H
#define _KMEAN_SEQ_H

#include <string.h>
#include <iostream>
#include <kmean.h>
#include <chrono>

class kmean_seq {
    
    private:      
    
    unsigned long int next;
    unsigned long   rmax;
    struct kmean_t *kmean;
    int rand(void);

    public:

    kmean_seq(struct kmean_t *km);
    ~kmean_seq(void);
    void set_seed(unsigned int sd); 
    void assign_RandomCentroids(void);
    void findNearestCentroids(void);
    void averageLabeledCentroids(void);
    float calc_EuclDist(float *pt1, float *pt2, int n_elem);
    float calc_Old2NewCentroidsDist(void);
 

};

#endif