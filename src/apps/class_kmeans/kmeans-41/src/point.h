#pragma once

#include <iostream>

struct point {
    point() {}
    point(std::string&& id, std::vector<double> values): id(std::move(id)), values(values) {}
    point(const point& p) {
        id = p.id;
        values = p.values;
    }
    point(point&& p) {
        id = std::move(p.id);
        values = std::move(p.values);
    }
//     point(std::string&& id, std::vector<double>&& values): id(std::move(id)), values(std::move(values)) {}
    void print() {
        std::cout << "ID " << id << " [";
        for (int i = 0; i < (int)values.size(); i++) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << values[i];
        }
        std::cout << "]" << std::endl;
    }
    
    std::string id;
    std::vector<double> values;
};