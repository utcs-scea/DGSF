#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <chrono>

const uint64_t MAX_THREADS = 256;
#define BLK(n) min((int)ceil((float)n / MAX_THREADS), 32768)
#define THREAD(n) min(n, MAX_THREADS)

__global__ void sum(int* dest, size_t n, int val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < n; i += blockDim.x * gridDim.x) {
       dest[i] += val;
    }
}

int main(int argc, char* argv[]) {
    
    if (argc != 2) {
        printf("need arg\n");
        return 0;
    }

    int64_t input_mbs = atoi(argv[1]);
    uint64_t bytes = input_mbs*1024*1024;
    uint64_t n = bytes/sizeof(int);

    auto start = std::chrono::steady_clock::now();
    auto istart = std::chrono::steady_clock::now();

    cudaFree(0);
    auto iend = std::chrono::steady_clock::now();

    printf("Allocating %d MBs\n", input_mbs);
    printf("  thats %lu bytes\n", input_mbs*1024*1024);
    int *d_a;
    //int *d_b;
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, bytes);
    printf("cudaMalloc returned: %d: %s\n", err, cudaGetErrorString(err));
    //printf("ptr is %p\n", d_a);
    err = cudaMemset((void*)d_a, 0, bytes);
    printf("cudaMemset returned: %d: %s\n", err, cudaGetErrorString(err));

    //printf("Launching %d   %d\n", BLK(n), THREAD(n));
    sum<<<BLK(n), THREAD(n)>>>(d_a, n, 1);
    //cudaDeviceSynchronize();

    // int buffer[10];
    // err = cudaMemcpy((void*)buffer, d_a, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("first 10 ints of 2nd array (should be 2): \n");
    // for (int i = 0 ; i < 10 ; i++)
    //     printf("%#02x ", ((int*)buffer)[i]);
    // printf("\n");

    sum<<<BLK(n), THREAD(n)>>>(d_a, n, 1);
    cudaDeviceSynchronize();

    int buffer[10];
    err = cudaMemcpy((void*)buffer, d_a, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("first 10 ints of 2nd array (should be 2): \n");
    for (int i = 0 ; i < 10 ; i++)
        printf("%#02x ", ((int*)buffer)[i]);
    printf("\n");

    auto end = std::chrono::steady_clock::now();

    printf(">>>> end to end %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    printf("init %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(iend - istart).count());
    
    cudaFree(d_a);

    return 0;
}
