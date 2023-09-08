#include <stdio.h>
 
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

__global__ void dummyKernel(float* data) {
    return;
}

int main(int argc, char* argv[]) {

    float *gpu1data;

    // Enable peer access
    cudaSetDevice(1);
    cudaMalloc(&gpu1data, 1000);

    printf("Enabling peer access on device 0 to device 1\n");

    cudaSetDevice(0);
    CHECK(cudaDeviceEnablePeerAccess(1,0));
    cudaPointerAttributes attr;

    for (int i = 0; i < 3; i++) {
        cudaMemset(gpu1data, 1, 1000);
        cudaError_t status = cudaPointerGetAttributes(&attr, gpu1data);
        printf("\ncudaPointerGetAttr returned: %d\n", status);
        printf("\n[Attributes]\n");
        printf("device: %d\n", attr.device);
        printf("devicePointer: %p\n", attr.devicePointer);
        printf("hostPointer: %p\n", attr.hostPointer);
    }

    printf("\n*** Done ***\n");
}
