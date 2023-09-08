#pragma once

#include <vector>
#include <fstream>
#include "point.h"

class loader {
    public:
    loader(int dims);
    std::vector<point> load(char* filename);
    std::vector<double> load_as_1d(char* filename);
    void load_as_pointer(char* filename, double** ptr, int& npoints);
    
    private:
    int dims_;
};