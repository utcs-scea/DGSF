#include <cuda.h>
#include <iostream>



int main() {
    /*
    cudaSetDevice(0);
    int *d_a;
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, 16*1024);
    std::cout << "return of cudaMalloc: " << err << std::endl;

    cudaPointerAttributes at;
    err = cudaPointerGetAttributes(&at, d_a);
    printf("d_a attr ret %d  type %d  device  %d    dvcptr %p hostptr %p\n", err, at.type, at.device, at.devicePointer, at.hostPointer);

    cudaSetDevice(1);

    err = cudaPointerGetAttributes(&at, d_a);
    printf("d_a attr ret %d  type %d  device  %d    dvcptr %p hostptr %p\n", err, at.type, at.device, at.devicePointer, at.hostPointer);
*/

    /*    
    CUresult r;

    cuInit(0);

    std::cout << "!!!  creating context on device 0!" << std::endl;

    // Get handle for device 0
    CUdevice cuDevice;
    r = cuDeviceGet(&cuDevice, 0);
    std::cout << "return of cuDeviceGet: " << r << std::endl;

    // Create context
    CUcontext cuContext;
    r = cuCtxCreate(&cuContext, 0, cuDevice);
    std::cout << "return of cuCtxCreate: " << r << std::endl;

    // Create module from binary file
    CUmodule cuModule;
    r = cuModuleLoad(&cuModule, "minimal.ptx");
    std::cout << "return of cuModuleLoad: " << r << std::endl;

    CUfunction sum;
    r = cuModuleGetFunction(&sum, cuModule, "_Z3sumv");
    std::cout << "return of cuModuleGetFunction: " << r << std::endl;

    r = cuLaunchKernel(sum, 1, 1, 1, 4, 1, 1, 0, 0, 0, 0);
    std::cout << "return of cuLaunchKernel: " << r << std::endl;

    std::cout << "!!!  switching context to device 1!" << std::endl;

    r = cuDeviceGet(&cuDevice, 1);
    std::cout << "return of cuDeviceGet: " << r << std::endl;

    r = cuCtxCreate(&cuContext, 0, cuDevice);
    std::cout << "return of cuCtxCreate: " << r << std::endl;

    r = cuLaunchKernel(sum, 1, 1, 1, 4, 1, 1, 0, 0, 0, 0);
    std::cout << "return of cuLaunchKernel: " << r << std::endl;
*/

    return 0;


}
