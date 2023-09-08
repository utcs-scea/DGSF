#include <stdio.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>




#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>

class abc {
	public:
	int _index;
	thrust::device_vector<double>::iterator _a;
};



void check(thrust::device_vector<double>& a) {

	std::cout << a[0] << " " << a[1] << "\n";
	a[0]=23;
}


 struct saxpy {
     const double _a;
     saxpy(int a) : _a(a) { }

     __host__ __device__
     double operator()(const double &x, const double& y) const {
         return _a * x + y;
     }
 };

 struct saxpy2 {
 double *a;
 saxpy2(double *a): a(a) {}
 

     __host__ __device__
     double operator()(const thrust::device_ptr<double> &x, const thrust::device_ptr<double>& y) const {
         printf("chetan test%f  %f %f\n", a[0], a[1], a[3]);
         return *x + *y;
     }
 };



int main(void)
{
 // H has storage for 4 integers
 thrust::host_vector<int> H(4);
 // initialize individual elements
 H[0] = 14;
 H[1] = 20;
 H[2] = 38;
 H[3] = 46;

 // H.size() returns the size of vector H
 std::cout << "H has size " << H.size() << std::endl;
 // print contents of H
 for(int i = 0; i < H.size(); i++)
 std::cout << "H[" << i << "] = " << H[i] << std::endl;
 // resize H
 H.resize(2);

 std::cout << "H now has size " << H.size() << std::endl;
 // Copy host_vector H to device_vector D
 thrust::device_vector<int> D = H;

 // elements of D can be modified
 D[0] = 99;
 D[1] = 88;

 // print contents of D
 for(int i = 0; i < D.size(); i++)
 std::cout << "D[" << i << "] = " << D[i] << std::endl;
 // H and D are automatically deleted when the function returns
 
 
 thrust::device_vector<double> data_points(6);
 data_points[0]= 1;
 data_points[1] = 2;
 data_points[2]= 3;
 data_points[3] = 4;
 data_points[4]= 5;
 data_points[5] = 6;

 thrust::device_vector<double> key_points(6);
 key_points[0]= 0;
 key_points[1] = 0;
 key_points[2]= 1;
 key_points[3] = 1;
 key_points[4]= 2;
 key_points[5] = 2;


 thrust::device_vector<double> centroid_points(4);
 centroid_points[0]= 1;
 centroid_points[1] = 2;
 centroid_points[2]= 3;
 centroid_points[3] = 4;

 thrust::device_vector<abc> d_points(3);
 //d_points[0]._a = data_points.begin();
 //d_points[1]._a = d_points[0]._a + 2;
 //d_points[2]._a = d_points[1]._a + 2;
 

 thrust::device_vector<double> result_points(6);

 thrust::transform(data_points.begin(), data_points.end(), key_points.begin(), result_points.begin(), saxpy(1));

 
 check(data_points); 
 std::cout << "\n" << result_points[2] << " " << result_points[5] << "\n";


 thrust::device_ptr<double> dev_ptr = thrust::device_malloc<double>(6);
 dev_ptr[0] = 10;
 dev_ptr[1] = 20;
 dev_ptr[2]  = 30;
  thrust::device_vector<thrust::device_ptr<double>> f_points(3);
 f_points[0] = dev_ptr;
 f_points[1] = dev_ptr + 2;
 f_points[2] = dev_ptr + 4;

 thrust::device_ptr<double> dev_ptr2 = thrust::device_malloc<double>(6);
 dev_ptr2[0] = 10;
 dev_ptr2[1] = 20;
 dev_ptr2[2]  = 30;
  thrust::device_vector<thrust::device_ptr<double>> f_points2(3);
 f_points2[0] = dev_ptr2;
 f_points2[1] = dev_ptr2 + 2;
 f_points2[2] = dev_ptr2 + 4;


 thrust::device_vector<double> result_points2(3);

 thrust::transform(f_points.begin(), f_points.end(), f_points2.begin(), result_points2.begin(),
 saxpy2(thrust::raw_pointer_cast(dev_ptr)));



 std::cout << result_points2[0] << " "  << result_points2[1] << " " << result_points2[2];
 
 
 thrust::device_vector<int> x(3);
 x[0] = 3;
 x[1] = 2;
 x[2]  = 1;
 
 
 
  thrust::device_vector<int> y(3);
  y[0] = 1;
  y[1] = 2;
  y[2] = 3;
  
  
    thrust::device_vector<int> z(3);
  z[0] = 10;
  z[1] = 13;
  z[2] = 11;
  
  
          // typedef these iterators for shorthand
typedef thrust::device_vector<int>::iterator   IntIterator;

// typedef a tuple of these iterators
typedef thrust::tuple<IntIterator, IntIterator> IteratorTuple;
// typedef the zip_iterator of this tuple
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
// finally, create the zip_iterator
ZipIterator iter(thrust::make_tuple(y.begin(), z.begin()));

thrust::stable_sort_by_key(x.begin(), x.end(), iter);


std::cout << std::endl;
std::cout << x[0] << " " << x[1] << std::endl
<< y[0] << " " << y[1] << std::endl
<< z[0] << " " << z[1];


std::cout << std::endl;
 
 
 return 0;
}

