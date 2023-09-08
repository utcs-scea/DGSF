#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include "io.h"
#include "kmeans.h"

using namespace std;

// reference: https://github.com/NVIDIA/thrust/blob/main/examples/expand.cu
// This example demonstrates how to expand an input sequence by 
// replicating each element a variable number of times. For example,
//
//   expand([2,2,2],[A,B,C]) -> [A,A,B,B,C,C]
//   expand([3,0,1],[A,B,C]) -> [A,A,A,C]
//   expand([1,3,2],[A,B,C]) -> [A,B,B,B,C,C]
//
// The element counts are assumed to be non-negative integers

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


// reference: https://github.com/NVIDIA/thrust/blob/main/examples/tiled_range.cu
// this example illustrates how to tile a range multiple times
// examples:
//   tiled_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3] 
//   tiled_range([0, 1, 2, 3], 2) -> [0, 1, 2, 3, 0, 1, 2, 3] 
//   tiled_range([0, 1, 2, 3], 3) -> [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] 
//   ...

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


// reference: https://github.com/NVIDIA/thrust/blob/main/examples/repeated_range.cu
// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3] 
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] 
//   ...

template <typename Iterator>
class repeated_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type repeats;

        repeat_functor(difference_type repeats)
            : repeats(repeats) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i / repeats;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<repeat_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the repeated_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    repeated_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
    }

    iterator end(void) const
    {
        return begin() + repeats * (last - first);
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type repeats;
};


// reference: https://github.com/NVIDIA/thrust/blob/main/examples/strided_range.cu
// this example illustrates how to make strided access to a range of values
// examples:
//   strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6] 
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
//   ...

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};


template <typename Vector>
void print(const std::string& s, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << s;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

static unsigned long int next_rand = 1;
static unsigned long kmeans_rmax = 32767;
void kmeans_srand(unsigned int seed) {
    next_rand = seed;
}
int kmeans_rand() {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand/65536) % (kmeans_rmax+1);
}


void init_centroids(struct options_t* args,
                    REAL*           dataset,
                    REAL*           centroids) {

    for (int k = 0; k < args->num_cluster; k++) {
        int index = kmeans_rand() % args->num_points;
        for (int d = 0; d < args->dims; d++) {
            centroids[k*args->dims+d] = dataset[index*args->dims+d];
        }
    }

}

struct calc_diff_sqr{
    __host__ __device__
    REAL operator()(REAL x, REAL y){
        return (x-y) * (x-y);
    }
};

struct arg_min{
    __host__ __device__
    thrust::tuple<int,REAL> operator()(const thrust::tuple<int,REAL>& a, const thrust::tuple<int,REAL>& b) const {
        if (thrust::get<1>(a) < thrust::get<1>(b)){
            return a;
        } else {
            return b;
        }
    }
};

struct copy_argmin{
    int num_cluster;

    copy_argmin(int k){
        num_cluster = k;
    }

    __host__ __device__
    int operator()(const thrust::tuple<int,REAL>& a) const {
        return thrust::get<0>(a) % num_cluster;
    }
};

struct check_convergence{
    REAL threshold;
    check_convergence(REAL t){ threshold = t; }
    __host__ __device__
    bool operator()(REAL x){
        if( abs(x) > threshold ){
            return 1;
        }else{
            return 0;
        }
    }
};


void kmeans_thrust(struct options_t* args,
                   REAL**            dataset,
                   int**             labels,
                   REAL**            centroids,
                   float*            time_loops, 
                   int*              iter_to_converge) {

    // initialize centroids randomly
    kmeans_srand(args->seed);
    init_centroids(args, *dataset, *centroids);
    thrust::host_vector<int>    index_host(args->num_points*args->dims*args->num_cluster);
    for(int p = 0; p < args->num_points; p++){
        for(int d = 0; d < args->dims; d++){
            for(int k = 0; k < args->num_cluster; k++){
                index_host[p*(args->num_cluster*args->dims)+d*args->num_cluster+k] = p*args->num_cluster + k;
            }
        }
    }

    // transfer to device
    thrust::device_vector<REAL>   dataset_dev(*dataset, *dataset + args->num_points*args->dims);
    thrust::device_vector<int>    labels_dev(*labels, *labels + args->num_points);
    thrust::device_vector<REAL>   centroids_dev(*centroids, *centroids + args->num_cluster*args->dims);
    thrust::device_vector<REAL>   old_centroids_dev(args->num_cluster*args->dims);
    thrust::device_vector<REAL>   centroids_diff_dev(args->num_cluster*args->dims);

    thrust::device_vector<REAL>   dataset_long_dev(args->num_points*args->dims*args->num_cluster);
    thrust::device_vector<int>    index_dev(args->num_points*args->dims*args->num_cluster);
    index_dev = index_host;
    thrust::device_vector<int>    index_tmp_dev(args->num_points*args->dims*args->num_cluster);
    thrust::device_vector<int>    index_reduce_dev(args->num_points*args->num_cluster);
    thrust::device_vector<int>    d_counts(args->num_points*args->dims, args->num_cluster);
    thrust::device_vector<int>    d_counts2(args->num_points, args->dims);
    thrust::device_vector<REAL>   diff_sqr_dev(args->num_cluster*args->dims*args->num_points);
    thrust::device_vector<REAL>   dist_dev(args->num_cluster*args->num_points);
    thrust::device_vector<thrust::tuple<int,REAL>> nearest_dev(args->num_points);

    thrust::device_vector<int>    labels_long_dev(args->num_points*args->dims);
    thrust::device_vector<int>    label_sizes_dev(args->num_cluster);
    thrust::device_vector<int>    label_sindex_dev(args->num_cluster);
    thrust::device_vector<REAL>   sum_centroids_dev(args->num_cluster);

    //print("datain index: ", index_dev);
    typedef thrust::device_vector<int>::iterator Iteratori;
    typedef thrust::device_vector<REAL>::iterator Iterator;
    //cout << "size: " << dataset_dev.size() * args->num_cluster << " : " << centroids_dev.size() * args->num_points;

    // core algorithm
    int iterations = 0;
    bool done = false;
    auto start = std::chrono::high_resolution_clock::now();
    while(!done) {

        thrust::copy(centroids_dev.begin(), centroids_dev.end(), old_centroids_dev.begin());
        thrust::copy(index_dev.begin(), index_dev.end(), index_tmp_dev.begin());
        iterations++;

        // labels is a mapping from each point in the dataset 
        // to the nearest (euclidean distance) centroid
        expand(d_counts.begin(), d_counts.end(), dataset_dev.begin(), dataset_long_dev.begin());
        thrust::stable_sort_by_key(index_tmp_dev.begin(), index_tmp_dev.end(), dataset_long_dev.begin());
        tiled_range<Iterator> centroids_long(centroids_dev.begin(), centroids_dev.end(), args->num_points);
        thrust::transform(dataset_long_dev.begin(), dataset_long_dev.end(), centroids_long.begin(), diff_sqr_dev.begin(), calc_diff_sqr());
        thrust::reduce_by_key(index_tmp_dev.begin(), index_tmp_dev.end(), diff_sqr_dev.begin(), index_reduce_dev.begin(), dist_dev.begin());
        repeated_range<Iteratori> index_rep(index_reduce_dev.begin(), index_reduce_dev.begin()+args->num_points, args->num_cluster);
        thrust::pair<thrust::device_vector<int>::iterator,thrust::device_vector<REAL>::iterator> new_end;
        thrust::reduce_by_key(index_rep.begin(), index_rep.end(), thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), dist_dev.begin())),
            index_reduce_dev.begin(), nearest_dev.begin(), thrust::equal_to<int>(), arg_min());
        thrust::transform(nearest_dev.begin(), nearest_dev.end(), labels_dev.begin(), copy_argmin(args->num_cluster));
        //print("labels thrust: ", labels_dev);
        //print("reduce index: ", index_reduce_dev);

        

        // the new centroids are the average 
        // of all the points that map to each 
        // centroid
        expand(d_counts2.begin(), d_counts2.end(), labels_dev.begin(), labels_long_dev.begin());
        thrust::stable_sort_by_key(labels_long_dev.begin(), labels_long_dev.end(), dataset_dev.begin());
        thrust::sort(labels_dev.begin(), labels_dev.end());
        for(int d = 0; d < args->dims; d++){
            strided_range<Iterator> data_dim_rep(dataset_dev.begin()+d, dataset_dev.end(), args->dims);
            thrust::reduce_by_key(labels_dev.begin(), labels_dev.end(), data_dim_rep.begin(), index_tmp_dev.begin(), sum_centroids_dev.begin());
            thrust::host_vector<int>   h_map(args->num_cluster);
            for(int k = 0; k < args->num_cluster; k++){
                h_map[k] = k*args->dims+d;
                //cout << h_map[k] << " ";
            }
            thrust::device_vector<int> d_map(args->num_cluster);
            d_map = h_map;
            thrust::scatter(sum_centroids_dev.begin(), sum_centroids_dev.end(),
                            d_map.begin(), centroids_dev.begin());
        }
        //print("sum centroids: ", centroids_dev);
        thrust::sort(labels_dev.begin(), labels_dev.end());
        thrust::reduce_by_key(labels_dev.begin(), labels_dev.end(), thrust::constant_iterator<int>(1), label_sindex_dev.begin(), label_sizes_dev.begin());
        repeated_range<Iteratori> sizes_rep(label_sizes_dev.begin(), label_sizes_dev.end(), args->dims);
        thrust::transform(centroids_dev.begin(), centroids_dev.end(), sizes_rep.begin(), centroids_dev.begin(), thrust::divides<REAL>());
        //print("centroids: ", centroids_dev);

        thrust::transform(centroids_dev.begin(), centroids_dev.end(), old_centroids_dev.begin(), centroids_diff_dev.begin(), thrust::minus<REAL>());
        int violate_count = thrust::transform_reduce(centroids_diff_dev.begin(), centroids_diff_dev.end(), check_convergence(args->threshold), 0, thrust::maximum<int>());
        done = iterations > args->max_num_iter || violate_count == 0;

    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    *time_loops = diff.count()/1000.0;
    *iter_to_converge = iterations;

    thrust::copy(labels_dev.begin(), labels_dev.end(), *labels);
    thrust::copy(centroids_dev.begin(), centroids_dev.end(), *centroids);

    return;

} 

