#include <iostream>
#include <chrono>
#include "argparse.h"
#include "loader.h"
#include "kmeans.h"

int main(int argc, char** argv) {
    options_t opts;
    get_opts(argc, argv, &opts);
    loader file_loader(opts.dims);
    
    auto points = file_loader.load(opts.inputfilename);
    
    kmeans impl(
        opts.num_cluster,
        opts.dims,
        std::move(points),
        std::make_unique<rng>(opts.seed),
        opts.max_num_iter,
        opts.threshold);
    impl.find();
    impl.print_time();
    if (opts.output_centroids) {
        impl.print_centroids();
    } else {
        impl.print_labels();
    }
}