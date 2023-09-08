#include "kmeans.h"

kmeans::kmeans(
    int k,
    int dims,
    std::vector<point>&& dataset,
    std::unique_ptr<rng> randomizer,
    int max_iterations,
    double threshold) : k_(k), dims_(dims), dataset_(dataset), randomizer_(std::move(randomizer)), max_iterations_(max_iterations), convergence_threshold_(threshold), iteration_(0) {
    for (int i = 0; i < k_; i++) {
        int index = randomizer_->kmeans_rand() % dataset.size();
        centroids_.emplace_back("centroid_" + std::to_string(i), dataset[index].values);
    }
}

void kmeans::find() {
    start_time_ = std::chrono::high_resolution_clock::now();
    while (iteration_ < max_iterations_) {
        iteration_++;
        std::vector<int> labels;
        for (const auto& p: dataset_) {
            labels.push_back(nearest_centroid_(p));
        }

        auto previous_centroids = centroids_;
        centroids_ = generate_new_centroids_(labels);
        
        if (is_converged_(previous_centroids)) {
            break;
        }
    }
    end_time_ = std::chrono::high_resolution_clock::now();
}

void kmeans::print_centroids() {
    for (int i = 0; i < k_; i++) {
        printf("%d ", i);
        for (int j = 0; j < dims_; j++) {
            printf("%lf ", centroids_[i].values[j]);
        }
        printf("\n");
    }
}

void kmeans::print_labels() {
    printf("clusters:");
    for (int p = 0; p < (int)dataset_.size(); p++) {
        printf(" %d", nearest_centroid_(dataset_[p]));
    }
}

void kmeans::print_time() {
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
    printf("%d,%ld\n", iteration_, diff.count());
}

std::vector<point> kmeans::generate_new_centroids_(const std::vector<int>& labels) {
    std::vector<int> count(k_, 0);
    std::vector<point> new_centroids;
    for (int i = 0; i < k_; i++) {
        new_centroids.emplace_back("", std::vector<double>(dims_, 0));
    }
    
    for (int i = 0; i < (int)labels.size(); i++) {
        for (int j = 0; j < dims_; j++) {
            new_centroids[labels[i]].values[j] += dataset_[i].values[j];
        }
        count[labels[i]]++;
    }
    
    for (int i = 0; i < k_; i++) {
        for (int j = 0; j < dims_; j++) {
            new_centroids[i].values[j] /= count[i];
        }
    }
    
    return new_centroids;
}

int kmeans::nearest_centroid_(const point& p) {
    double min_distance = std::numeric_limits<double>::max();
    int selected = -1;
    for (int i = 0; i < k_; i++) {
        double distance = distance_(centroids_[i], p);
        if (distance < min_distance) {
            min_distance = distance;
            selected = i;
        }
    }
    return selected;
}

double kmeans::distance_(const point& a, const point& b) {
    double sum = 0;
    for (int i = 0; i < dims_; i++) {
        double d = a.values[i] - b.values[i];
        sum += d * d;
    }
    return sqrt(sum);
}

bool kmeans::is_converged_(const std::vector<point>& previous_centroids) {
    for (int i = 0; i < k_; i++) {
        if (distance_(centroids_[i], previous_centroids[i]) > convergence_threshold_) {
            return false;
        }
    }
    
    return true;
}