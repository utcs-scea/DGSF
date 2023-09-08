#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

/*
__global__ void sum(int* dest, size_t n, int val) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
       dest[tid] += val;
    }
}
*/

__global__ void sum() {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 10) {

    }
}

int main(int argc, char* argv[]) {
    int *d_a;
    int *d_b;
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, 16*1024*1024);
    printf("16MB cudaMalloc returned: %d: %s\n", err, cudaGetErrorString(err));
    printf("ptr is %p\n", d_a);

    err = cudaMemset((void*)d_a, 0, 4096 * sizeof(int));
    printf("cudaMemset returned: %d: %s\n", err, cudaGetErrorString(err));
    
    //sum<<<1, 64>>>(d_a, 64, 1);
    sum<<<1, 64>>>();
    cudaDeviceSynchronize();

    char buf[4096 * sizeof(int)];
    err = cudaMemcpy((void*)buf, d_a, 4096 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy returned: %d: %s\n", err, cudaGetErrorString(err));

    printf("first 10 ints after 0+1: \n");
    for (int i = 0 ; i < 10 ; i++)
        printf("%#02x ", ((int*)buf)[i]);
    
    err = cudaMalloc((void**)&d_b, 16*1024*1024);
    printf("\n2nd 16MB cudaMalloc returned: %d: %s\n", err, cudaGetErrorString(err));

    err = cudaMemset((void*)d_b, 0, 64 * sizeof(int));
    printf("cudaMemset returned: %d: %s\n", err, cudaGetErrorString(err));

    sum<<<1, 64>>>();
    cudaDeviceSynchronize();

    sum<<<1, 64>>>();
    cudaDeviceSynchronize();

    err = cudaMemcpy((void*)buf, d_b, 64 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy returned: %d: %s\n", err, cudaGetErrorString(err));

    printf("first 10 ints of 2nd array (should be 1): \n");
    for (int i = 0 ; i < 10 ; i++)
        printf("%#02x ", ((int*)buf)[i]);

    cudaDeviceSynchronize();
    return 0;
}
