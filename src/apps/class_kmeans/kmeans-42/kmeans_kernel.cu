#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <list>
#include "kmeans.h"
#include <iostream>
#include "cuda.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <assert.h>

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void
findNearestCentroid(const double *d_input_vec, const double *d_centroids, 
                    double *d_labels, int n_vals, int dims, int num_centroids)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n_vals) return;
    
    // Find nearest centroid for each index
    int min_index = 0;
    double min_distance = 0; 
    for (int j=0; j < num_centroids; j++){
        // calculate distance and find minimum distance
        double total = 0;
        for(int i=0; i < dims; i++){
            total += powf(d_input_vec[index*dims + i] - d_centroids[j*dims + i], 2);
        }
        double new_distance = powf(total, 0.5);
        if (j == 0) {
            min_distance = new_distance;
        }
        else if (new_distance <= min_distance){
            min_index = j;
            min_distance = new_distance;
        }
    }
    d_labels[index] = min_index;
}

__global__ void
getLabelCounts(const double *labels, int n_vals, double *counts)
{ 
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < n_vals){
        int c_index = labels[index];
        myAtomicAdd(&(counts[c_index]), 1.0);
    }
}

__global__ void
calculateNewCentroids(const double *d_input_vec, double *d_centroids, 
                      const double *labels, const double *counts, 
                      int n_vals, int dims)
{ 
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n_vals) return;
    
    int c_index = labels[index]; // centroid index
    // add all dims of particular point to centroids
    for (int i = 0; i < dims; i++) {
        myAtomicAdd(&(d_centroids[c_index*dims + i]), d_input_vec[index*dims + i]/counts[c_index]);
        
    }
}

__global__ void
testThreshold(const double *old_centroids, const double *new_centroids, int num_elements, double threshold, int *notChanged)
{ 
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < num_elements){
        double diff = fabsf(old_centroids[index] - new_centroids[index]);
        
        if (diff > threshold) {
            notChanged[0] = 1;
        }
    }
}



void cuda_basic_kmeans(int n_vals, int dims, int num_centroids, double *centroids, double *input_vals, double threshold, int max_num_iter, bool c, int threadsPerBlock){
    // Initialize timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *h_labels = (double *)malloc(n_vals*sizeof(double));
    double *h_counts = (double *)malloc(num_centroids*sizeof(double));
    // Initialize counts to 0
    for (int i = 0; i < num_centroids; ++i)
    {
        h_counts[i] = 0;
    }
    double *h_centroids = (double *)malloc(num_centroids*dims*sizeof(double));
    int *h_threshold_not_changed = (int *)malloc(sizeof(int));
    h_threshold_not_changed[0] = 0; // 0 -> true , 1 -> false
    
    double *d_input_vec = NULL;
    cudaMalloc((void **)&d_input_vec, n_vals*dims*sizeof(double));
    cudaMemcpy(d_input_vec, input_vals, n_vals*dims*sizeof(double), cudaMemcpyHostToDevice);

    double *d_centroids = NULL;
    cudaMalloc((void **)&d_centroids, num_centroids*dims*sizeof(double));
    cudaMemcpy(d_centroids, centroids, num_centroids*dims*sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_centroids_old = NULL;
    cudaMalloc((void **)&d_centroids_old, num_centroids*dims*sizeof(double));
    
    double *d_labels = NULL;
    cudaMalloc((void **)&d_labels, n_vals*sizeof(double));
    
    double *d_counts = NULL;
    cudaMalloc((void **)&d_counts, num_centroids*sizeof(double));
    cudaMemcpy(d_counts, h_counts, num_centroids*sizeof(double), cudaMemcpyHostToDevice);
    
    int *d_threshold_not_changed = NULL;
    cudaMalloc((void **)&d_threshold_not_changed, sizeof(int));
    cudaMemcpy(d_threshold_not_changed, h_threshold_not_changed, sizeof(int), cudaMemcpyHostToDevice);
    
    int blocksPerGrid = (n_vals + threadsPerBlock - 1) / threadsPerBlock;
    int iter_to_converge = 1;
    
    cudaEventRecord(start);
    for(int i = 0; i < max_num_iter; i++) {
        cudaMemcpy(d_centroids_old, d_centroids, num_centroids*dims*sizeof(double), cudaMemcpyDeviceToDevice);
        
        findNearestCentroid<<<blocksPerGrid, threadsPerBlock>>>(d_input_vec, d_centroids, d_labels, n_vals, dims, num_centroids);
        getLabelCounts<<<blocksPerGrid, threadsPerBlock>>>(d_labels, n_vals, d_counts);
        // Set all values in d_centroids to 0
        cudaMemset(d_centroids, 0, num_centroids*dims*sizeof(double));
        
        calculateNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_input_vec, d_centroids, d_labels, d_counts, n_vals, dims);
        
        // Set all values in d_counts to 0
        cudaMemset(d_counts, 0, num_centroids*sizeof(double));
        
        cudaMemset(d_threshold_not_changed, 0, sizeof(int));
        testThreshold<<<blocksPerGrid, threadsPerBlock>>>(d_centroids_old, d_centroids, num_centroids*dims, threshold, d_threshold_not_changed);
        
        cudaMemcpy(h_threshold_not_changed, d_threshold_not_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_threshold_not_changed[0] == 0) {
            break;
        }
        
        iter_to_converge++;
    }
     
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%d,%lf\n", iter_to_converge, milliseconds/iter_to_converge);
    
    if(c == true){
        cudaMemcpy(h_centroids, d_centroids, num_centroids*dims*sizeof(double), cudaMemcpyDeviceToHost);
        // print out centroids
        for(int i = 0; i < num_centroids; i++){
            printf("%d ", i);
            for (int j = 0; j < dims; j++)
                printf("%lf ", h_centroids[(i * dims) + j]);
            printf("\n");
        }
    } else {
        cudaMemcpy(h_labels, d_labels, n_vals*sizeof(double), cudaMemcpyDeviceToHost);
        // print cluster id of points
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", (int)h_labels[p]);
    }
    
    
}





///////////////////////////////// CUDA Shared Memory Implemention ////////////////////////////////////////////////////////////


__global__ void
findNearestCentroid_shmem(const double *d_input_vec, const double *d_centroids, 
                    double *d_labels, int n_vals, int dims, int num_centroids)
{
    extern __shared__ double temp[];
    int gindex = blockDim.x * blockIdx.x + threadIdx.x;
    int lindex = threadIdx.x;
    
    // Read inputs into shared memory
    if (lindex < num_centroids){
        for (int i = 0; i < dims; i++) {
            temp[lindex*dims+i] = d_centroids[lindex*dims+i];
        }
    }
    if (gindex >= n_vals) return;

    __syncthreads();

    // Find nearest centroid for each index
    int min_index = 0;
    double min_distance = 0; 
    for (int j=0; j < num_centroids; j++){
        // calculate distance and find minimum distance
        double total = 0;
        for(int i=0; i < dims; i++){
            total += powf(d_input_vec[gindex*dims + i] - temp[j*dims + i], 2);
        }
        double new_distance = powf(total, 0.5);
        if (j == 0) {
            min_distance = new_distance;
        }
        else if (new_distance <= min_distance){
            min_index = j;
            min_distance = new_distance;
        }
        
    }
    d_labels[gindex] = min_index;
}

void cuda_shmem_kmeans(int n_vals, int dims, int num_centroids, double *centroids, double *input_vals, double threshold, int max_num_iter, bool c, int threadsPerBlock){
    
    // Initialize timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *h_labels = (double *)malloc(n_vals*sizeof(double));
    double *h_counts = (double *)malloc(num_centroids*sizeof(double));
    // Initialize counts to 0
    for (int i = 0; i < num_centroids; ++i)
    {
        h_counts[i] = 0;
    }
    double *h_centroids = (double *)malloc(num_centroids*dims*sizeof(double));
    int *h_threshold_not_changed = (int *)malloc(sizeof(int));
    h_threshold_not_changed[0] = 0; // 0 -> true , 1 -> false
    
    double *d_input_vec = NULL;
    cudaMalloc((void **)&d_input_vec, n_vals*dims*sizeof(double));
    cudaMemcpy(d_input_vec, input_vals, n_vals*dims*sizeof(double), cudaMemcpyHostToDevice);

    double *d_centroids = NULL;
    cudaMalloc((void **)&d_centroids, num_centroids*dims*sizeof(double));
    cudaMemcpy(d_centroids, centroids, num_centroids*dims*sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_centroids_old = NULL;
    cudaMalloc((void **)&d_centroids_old, num_centroids*dims*sizeof(double));
    
    double *d_labels = NULL;
    cudaMalloc((void **)&d_labels, n_vals*sizeof(double));
    
    double *d_counts = NULL;
    cudaMalloc((void **)&d_counts, num_centroids*sizeof(double));
    cudaMemcpy(d_counts, h_counts, num_centroids*sizeof(double), cudaMemcpyHostToDevice);
    
    int *d_threshold_not_changed = NULL;
    cudaMalloc((void **)&d_threshold_not_changed, sizeof(int));
    cudaMemcpy(d_threshold_not_changed, h_threshold_not_changed, sizeof(int), cudaMemcpyHostToDevice);
    
    int blocksPerGrid = (n_vals + threadsPerBlock - 1) / threadsPerBlock;
    int iter_to_converge = 1;
    
    cudaEventRecord(start);
    for(int i = 0; i < max_num_iter; i++) {
        cudaMemcpy(d_centroids_old, d_centroids, num_centroids*dims*sizeof(double), cudaMemcpyDeviceToDevice);

        findNearestCentroid_shmem<<<blocksPerGrid, threadsPerBlock, num_centroids*dims*sizeof(double)>>>(d_input_vec, d_centroids, d_labels, n_vals, dims, num_centroids);
        
        getLabelCounts<<<blocksPerGrid, threadsPerBlock>>>(d_labels, n_vals, d_counts);
        // Set all values in d_centroids to 0
        cudaMemset(d_centroids, 0, num_centroids*dims*sizeof(double));
        
        calculateNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_input_vec, d_centroids, d_labels, d_counts, n_vals, dims);
        
        // Set all values in d_counts to 0
        cudaMemset(d_counts, 0, num_centroids*sizeof(double));
        
        cudaMemset(d_threshold_not_changed, 0, sizeof(int));
        testThreshold<<<blocksPerGrid, threadsPerBlock>>>(d_centroids_old, d_centroids, num_centroids*dims, threshold, d_threshold_not_changed);
        
        cudaMemcpy(h_threshold_not_changed, d_threshold_not_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_threshold_not_changed[0] == 0) {
            break;
        }
        
        iter_to_converge++;
    }
     
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%d,%lf\n", iter_to_converge, milliseconds/iter_to_converge);
    
    if(c == true){
        cudaMemcpy(h_centroids, d_centroids, num_centroids*dims*sizeof(double), cudaMemcpyDeviceToHost);
        // print out centroids
        for(int i = 0; i < num_centroids; i++){
            printf("%d ", i);
            for (int j = 0; j < dims; j++)
                printf("%lf ", h_centroids[(i * dims) + j]);
            printf("\n");
        }
    } else {
        cudaMemcpy(h_labels, d_labels, n_vals*sizeof(double), cudaMemcpyDeviceToHost);
        // print cluster id of points
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", (int)h_labels[p]);
    }
}


/////////////////////////// Thrust Implemention ///////////////////////////////

// The following 'tiled_range' is from the thrust examples found at https://github.com/NVIDIA/thrust/blob/master/examples/tiled_range.cu
template <typename Iterator>
class tiled_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct tile_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type tile_size;

        tile_functor(difference_type tile_size)
            : tile_size(tile_size) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i % tile_size;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator>   TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
    }

    iterator end(void) const
    {
        return begin() + tiles * (last - first);
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};

// The following 'expand' is from the thrust examples found at https://github.com/NVIDIA/thrust/blob/master/examples/expand.cu
template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
  difference_type input_size  = thrust::distance(first1, last1);
  difference_type output_size = thrust::reduce(first1, last1);

  // scan the counts to obtain output offsets for each input element
  thrust::device_vector<difference_type> output_offsets(input_size, 0);
  thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

  // scatter the nonzero counts into their corresponding output positions
  thrust::device_vector<difference_type> output_indices(output_size, 0);
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(input_size),
     output_offsets.begin(),
     first1,
     output_indices.begin());

  // compute max-scan over the output indices, filling in the holes
  thrust::inclusive_scan
    (output_indices.begin(),
     output_indices.end(),
     output_indices.begin(),
     thrust::maximum<difference_type>());

  // gather input values according to index array (output = first2[output_indices])
  OutputIterator output_end = output; thrust::advance(output_end, output_size);
  thrust::gather(output_indices.begin(),
                 output_indices.end(),
                 first2,
                 output);

  // return output + output_size
  thrust::advance(output, output_size);
  return output;
}

typedef thrust::device_vector<double>::iterator Iterator;
typedef thrust::tuple<double,double> DoubleTuple;
typedef thrust::device_vector<double>::iterator DoubleIterator;
typedef thrust::tuple<DoubleIterator, DoubleIterator> DoubleIteratorTuple;
typedef thrust::zip_iterator<DoubleIteratorTuple> Double2Iterator;

struct TupleDistanceMinimum
{ 
  __device__ DoubleTuple operator()(const DoubleTuple &lhs, const DoubleTuple &rhs) 
  {
      if(thrust::get<1>(lhs) < thrust::get<1>(rhs)){
          return lhs;
      } 
      return rhs;
  }
};

struct TupleEquality
{ 
  __device__ bool operator()(const DoubleTuple &lhs, const DoubleTuple &rhs) 
  {
      if(thrust::get<0>(lhs) == thrust::get<0>(rhs) && thrust::get<1>(lhs) == thrust::get<1>(rhs)){
          return true;
      } 
      return false;
  }
};

struct CentroidLabelSorter 
{
  __device__ bool operator()(const DoubleTuple &lhs, const DoubleTuple &rhs) 
  {
      
      if(thrust::get<0>(lhs) < thrust::get<0>(rhs)){
          return true;
      } else if (thrust::get<0>(lhs) == thrust::get<0>(rhs)) {
          if(thrust::get<1>(lhs) < thrust::get<1>(rhs)){
              return true;
          }
      }
      return false;
  }
};

struct FirstTupleElement
{
    __device__ float operator()(const DoubleTuple& a) const
    {
        return thrust::get<0>(a);
    }
};

// 'which_row' comes from https://github.com/NVIDIA/thrust/blob/master/examples/scan_matrix_by_rows.cu
struct which_row : thrust::unary_function<int, int> {
    int row_length;

    __host__ __device__
    which_row(int row_length_) : row_length(row_length_) {}

    __host__ __device__
    int operator()(int idx) const {
    return idx / row_length;
    }
};

// 'abs_diff' comes from https://github.com/NVIDIA/thrust/blob/master/examples/max_abs_diff.cu
template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return fabsf(b - a);
    }
};


void parallel_thrust_kmeans(int n_vals, int dims, int num_centroids, double *centroids, double *input_vals, double threshold, int max_num_iter, bool c)
{   
    // Initialize timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Initialize host/device vectors
    std::list<double> stl_input_keys;
    std::list<double> stl_centroid_keys;
    std::list<double> stl_dims_list;
    for(int i = 0; i < num_centroids; i++){
        for(int j = 0; j < n_vals; j++){
            stl_input_keys.push_back(j);
        }
    }
    for(int i = 0; i < n_vals; i++){
        for(int j = 0; j < num_centroids; j++){
            stl_centroid_keys.push_back(j);
        }
    }
    for(int i = 0; i < dims; i++) {
        stl_dims_list.push_back(i);
    }
    thrust::device_vector<double> input_keys(stl_input_keys.begin(), stl_input_keys.end());
    thrust::device_vector<double> input_keys_copy(num_centroids*n_vals);
    thrust::device_vector<double> centroid_keys(stl_centroid_keys.begin(), stl_centroid_keys.end());
    thrust::device_vector<double> distance(n_vals*dims*num_centroids);
    
    // https://github.com/NVIDIA/thrust/blob/master/examples/scan_matrix_by_rows.cu
    thrust::counting_iterator<double> c_first(0);
    thrust::transform_iterator<which_row, thrust::counting_iterator<double> >
        t_first(c_first, which_row(dims));
    
    thrust::device_vector<double> keys(t_first, t_first + n_vals * dims * num_centroids);
    thrust::device_vector<double> centroid_vec_keys(num_centroids * dims * n_vals); // empty
    thrust::device_vector<double> centroid_vec_keys_copy(num_centroids * dims * n_vals); // empty
    thrust::device_vector<double> reduced_dist_keys(n_vals*num_centroids);
    thrust::device_vector<double> reduced_dist(n_vals*num_centroids);
    thrust::device_vector<double> centroid_keys_copy(n_vals*num_centroids);
    thrust::device_vector<double> centroid_labels_copy(n_vals);
    thrust::device_vector<double> val_keys(n_vals);
    thrust::device_vector<DoubleTuple> centroid_label_tuples(n_vals);
    thrust::equal_to<double> binary_pred;
    thrust::device_vector<DoubleTuple> sum_centroid_keys(num_centroids*dims);
    thrust::device_vector<double> sum_centroid_values(num_centroids*dims);
    thrust::device_vector<double> centroid_labels(n_vals);
    thrust::device_vector<double> ones(n_vals);
    thrust::fill(ones.begin(), ones.end(), 1);
    
    thrust::device_vector<double> centroid_count_keys(num_centroids);
    thrust::device_vector<double> centroid_counts(num_centroids);
    thrust::device_vector<double> centroid_labels_expanded(n_vals*dims);
    thrust::device_vector<double> input_vec_copy(n_vals*dims);
    thrust::device_vector<double> centroid_counts_expanded(num_centroids*dims);
    
    // Initialize points, centroids and output_vec
    thrust::device_vector<double> input_vec(input_vals, input_vals + (n_vals * dims));
    thrust::device_vector<double> centroid_vec(centroids, centroids + (num_centroids * dims));
    thrust::device_vector<double> centroid_vec_old(num_centroids * dims);
    thrust::device_vector<int> d_counts(n_vals*num_centroids);
    thrust::fill(d_counts.begin(), d_counts.end(), dims);
    
    // 'abs_diff' from https://github.com/NVIDIA/thrust/blob/master/examples/max_abs_diff.cu
    thrust::maximum<double> binary_op1;
    abs_diff<double>        binary_op2;
    
    thrust::device_vector<int> dim_counts(n_vals);
    thrust::fill(dim_counts.begin(), dim_counts.end(), dims);
    
    // expand values according to counts
    expand(d_counts.begin(), d_counts.end(), centroid_keys.begin(), centroid_vec_keys.begin());

    // create tiled_range with num points tiles
    tiled_range<Iterator> inputs_tiled(input_vec.begin(), input_vec.end(), num_centroids);
    thrust::device_vector<double> centroids_tiled_result(n_vals*num_centroids*dims);
    
    int iter_to_converge = 0;
    
    cudaEventRecord(start);
    
    for (int i = 0; i < max_num_iter; i++){
        tiled_range<Iterator> centroids_tiled(centroid_vec.begin(), centroid_vec.end(), n_vals); // changes each loop
        thrust::copy(centroid_vec.begin(), centroid_vec.end(), centroid_vec_old.begin());
        
        thrust::copy(centroids_tiled.begin(), centroids_tiled.end(), centroids_tiled_result.begin());
        thrust::copy(centroid_vec_keys.begin(), centroid_vec_keys.end(), centroid_vec_keys_copy.begin());
        // stable sort by key for centroids_tiled
        thrust::stable_sort_by_key(centroid_vec_keys_copy.begin(), centroid_vec_keys_copy.end(), centroids_tiled_result.begin());

        thrust::transform(inputs_tiled.begin(), inputs_tiled.end(), centroids_tiled_result.begin(), distance.begin(),
                      thrust::minus<double>());
        // square distances
        thrust::transform(distance.begin(), distance.end(), distance.begin(), thrust::square<double>());

        // add distances for each point
        thrust::reduce_by_key(keys.begin(), keys.end(), distance.begin(), reduced_dist_keys.begin(), reduced_dist.begin());

        // Copy keys 
        thrust::copy(centroid_keys.begin(), centroid_keys.end(), centroid_keys_copy.begin());
        thrust::sort(centroid_keys_copy.begin(), centroid_keys_copy.end());

        Double2Iterator first = thrust::make_zip_iterator(thrust::make_tuple(centroid_keys_copy.begin(), reduced_dist.begin()));

        thrust::copy(input_keys.begin(), input_keys.end(), input_keys_copy.begin());
        thrust::sort_by_key(input_keys_copy.begin(), input_keys_copy.end(), first);
        thrust::reduce_by_key(input_keys_copy.begin(), input_keys_copy.end(), first, val_keys.begin(), centroid_label_tuples.begin(), binary_pred, TupleDistanceMinimum());

        // Get centroid labels for each point
        thrust::transform(centroid_label_tuples.begin(), centroid_label_tuples.end(), centroid_labels.begin(), FirstTupleElement());
        thrust::copy(centroid_labels.begin(), centroid_labels.end(), centroid_labels_copy.begin());
        
        // Expand centroid_labels to dims
        expand(dim_counts.begin(), dim_counts.end(), centroid_labels.begin(), centroid_labels_expanded.begin());

        thrust::device_vector<double> dim_indices(dims);
        thrust::sequence(dim_indices.begin(), dim_indices.end());
        tiled_range<Iterator> dim_indices_tiled_iter(dim_indices.begin(), dim_indices.end(), n_vals);
        thrust::device_vector<double> dim_indices_tiled_vec(dims*n_vals);
        thrust::copy(dim_indices_tiled_iter.begin(), dim_indices_tiled_iter.end(), dim_indices_tiled_vec.begin());
        
        // zip together into tuples
        Double2Iterator centroid_dim_keys_start = thrust::make_zip_iterator(thrust::make_tuple(centroid_labels_expanded.begin(), dim_indices_tiled_vec.begin()));
        Double2Iterator centroid_dim_keys_end = thrust::make_zip_iterator(thrust::make_tuple(centroid_labels_expanded.end(), dim_indices_tiled_vec.end()));

        // Make copy of input_vec
        thrust::copy(input_vec.begin(), input_vec.end(), input_vec_copy.begin());
        thrust::stable_sort_by_key(centroid_dim_keys_start, centroid_dim_keys_end, input_vec_copy.begin(), CentroidLabelSorter());
        thrust::reduce_by_key(centroid_dim_keys_start, centroid_dim_keys_end, input_vec_copy.begin(), sum_centroid_keys.begin(), sum_centroid_values.begin(), TupleEquality());

        // Get count for each centroid 
        thrust::sort(centroid_labels_copy.begin(), centroid_labels_copy.end());
        thrust::reduce_by_key(centroid_labels_copy.begin(), centroid_labels_copy.end(), ones.begin(), centroid_count_keys.begin(), centroid_counts.begin());
        
        expand(d_counts.begin(), d_counts.begin() + num_centroids, centroid_counts.begin(), centroid_counts_expanded.begin());

        // Transform to get new average of points
        // compute Y = X / Z
        thrust::transform(sum_centroid_values.begin(), sum_centroid_values.end(), centroid_counts_expanded.begin(), centroid_vec.begin(), thrust::divides<double>());
        iter_to_converge++;
        // Calculate convergence
        double init = 0;
        
        double max_abs_diff = thrust::inner_product(centroid_vec_old.begin(), centroid_vec_old.end(), centroid_vec.begin(), init, binary_op1, binary_op2);
        if(max_abs_diff < threshold){
            break;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%d,%lf\n", iter_to_converge, milliseconds/iter_to_converge);
    
    if(c == true){
        // print centroids
        std::vector<double> stl_vector(num_centroids*dims);
        thrust::copy(centroid_vec.begin(), centroid_vec.end(), stl_vector.begin());
        for(int i = 0; i < num_centroids; i++){
            printf("%d ", i);
            for (int j = 0; j < dims; j++)
                printf("%lf ", stl_vector[(i * dims) + j]);
            printf("\n");
        }
    } else {
        // print cluster id of points
        std::vector<double> clusterId_of_point(n_vals);
        thrust::copy(centroid_labels.begin(), centroid_labels.end(), clusterId_of_point.begin());
        printf("clusters:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", (int)clusterId_of_point[p]);
    }
}
