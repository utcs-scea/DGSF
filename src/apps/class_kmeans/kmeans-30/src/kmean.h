#ifndef _KMEAN_H
#define _KMEAN_H

struct kmean_t {
    char 	     *in_file;
    int 	     n_points;
    int        n_clusters;
    int            n_dims;
    int        n_max_iter;
    float       f_thresh;
    bool       b_centroid;
    unsigned int   n_seed;
    float**    input_pts;
    int*       cluster_id;
    float**    centroids;
    float** old_centroids;
    int             n_rev;       // specify the program revision (Seq/Thrust/Cuda)
    
};



#endif