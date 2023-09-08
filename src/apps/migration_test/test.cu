#include <stdio.h>
#include <unistd.h>
#include <string.h>
 
#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
     printf("Error: %s:%d, ", __FILE__, __LINE__); \
     printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
     exit(1); \
 } \
}

#define DATA_SIZE 1000000

__global__ void accessMemory(float* data) {
    data[0] = 1.0;
}

__global__ void basicKernel(int* data) {
    for (int i = 0; i < DATA_SIZE; i++)
        data[i] += i * 3;
}

int main(int argc, char* argv[]) {

    if (argc > 1 && strcmp(argv[1], "-c") == 0) {
        int *data;
        cudaMalloc(&data, sizeof(int) * DATA_SIZE);
        while (true) {
            printf("launch kernel\n");
            basicKernel<<<1,1>>>(data);
            sleep(2);
            printf("done with kernel\n");
        }
    } 
    /*
    else {
        int access2from1, access1from2;
        cudaDeviceCanAccessPeer(&access2from1, 1, 0);
        cudaDeviceCanAccessPeer(&access1from2, 0, 1);

        bool sameComplex = false;
        if (access2from1 && access1from2) {
            sameComplex = true;
        }

        printf("Check access ability\n");
        if (sameComplex) {
            printf("Enabling peer access\n");
            // enable peer access
            int device;
            cudaSetDevice(1);
            cudaGetDevice(&device);
            printf("enabling peer access on device %d to device %d\n", 0, device);
            CHECK(cudaDeviceEnablePeerAccess(0,0));

            cudaSetDevice(0);
            cudaGetDevice(&device);
            printf("enabling peer access on device %d to device %d\n", 1, device);
            CHECK(cudaDeviceEnablePeerAccess(1,0));

            // allocate some data
            float *gpu1data, *gpu2data;
            cudaSetDevice(1);
            cudaMalloc(&gpu1data, 1000);
            cudaSetDevice(0);
            cudaMalloc(&gpu2data, 1000);

            accessMemory<<<1,1>>>(gpu1data);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(gpu1data, gpu2data, 1000, cudaMemcpyDeviceToDevice));
        }

    }
    */

    return 0;
}
