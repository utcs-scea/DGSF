#pragma once

void kcalcDistance(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims);

void knaiiveArgmin(const double* distances, int* labels, int n_points, int n_centroids);

void kcalcNewCentroids(const double* data, const int* labels, double* newCentroids, int* counts, int n_points, int dims, int n_centroids);

void kcalcDistanceShared(const double* data, const double* centroids, double* output, int n_points, int n_centroids, int dims);