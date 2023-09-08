#pragma once

class rng {
    public:
    rng(unsigned int seed);
    int kmeans_rand();
    
    private:
    unsigned long int next_;
    const unsigned long KMEANS_RMAX = 32767;
};