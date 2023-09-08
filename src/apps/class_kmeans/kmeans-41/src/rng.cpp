#include "rng.h"

rng::rng(unsigned int seed): next_(seed) {}
int rng::kmeans_rand() {
    next_ = next_ * 1103515245 + 12345;
    return (unsigned int)(next_/65536) % (KMEANS_RMAX+1);
}
