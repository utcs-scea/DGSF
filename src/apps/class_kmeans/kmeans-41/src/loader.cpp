#include "loader.h"

loader::loader(int dims): dims_(dims) {}

std::vector<point> loader::load(char* filename) {
    std::ifstream in;
    in.open(filename);
    
    int num_points;
    in >> num_points;
    
    std::vector<point> ret;
    for (int i = 0; i < num_points; i++) {
        std::string id;
        in >> id;
        
        std::vector<double> point(dims_);
        for (int j = 0; j < dims_; j++) {
            in >> point[j];
        }
        ret.emplace_back(std::move(id), std::move(point));
    }
    
    return ret;
}

std::vector<double> loader::load_as_1d(char* filename) {
    std::ifstream in;
    in.open(filename);
    
    int num_points;
    in >> num_points;
    
    std::vector<double> ret(dims_ * num_points);
    for (int i = 0; i < num_points; i++) {
        std::string id;
        in >> id;
        for (int j = 0; j < dims_; j++) {
            in >> ret[i * dims_ + j];
        }
    }
    
    return ret;
}

void loader::load_as_pointer(char* filename, double** ptr, int& npoints) {
    std::ifstream in;
    in.open(filename);
    
    in >> npoints;
    
    *ptr = (double*)malloc(npoints * dims_ * sizeof(double));
    for (int i = 0; i < npoints; i++) {
        std::string id;
        in >> id;
        for (int j = 0; j < dims_; j++) {
            in >> (*ptr)[i * dims_ + j];
        }
    }
}