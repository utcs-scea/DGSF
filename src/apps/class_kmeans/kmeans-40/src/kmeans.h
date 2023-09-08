#pragma once

#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <iostream>

void kmeans_seq(struct options_t* args,
                REAL**            dataset,
                int**             labels,
                REAL**            centroids,
                float*            time_loops, 
                int*              iter_to_converge);

void kmeans_thrust(struct options_t* args,
                   REAL**            dataset,
                   int**             labels,
                   REAL**            centroids,
                   float*            time_loops, 
                   int*              iter_to_converge);

void kmeans_cuda(struct options_t* args,
                 REAL**            dataset,
                 int**             labels,
                 REAL**            centroids,
                 float*            time_loops, 
                 int*              iter_to_converge);

void kmeans_cuda_shmem(struct options_t* args,
                       REAL**            dataset,
                       int**             labels,
                       REAL**            centroids,
                       float*            time_loops, 
                       int*              iter_to_converge);
