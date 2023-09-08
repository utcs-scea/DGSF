#include <cuda.h>
#include <iostream>



int main() {
    CUresult r;

    cuInit(0);

    std::cout << "!!!  creating context on device 0!" << std::endl;

    // Get handle for device 0
    CUdevice cuDevice;
    r = cuDeviceGet(&cuDevice, 0);
    std::cout << "return of cuDeviceGet: " << r << std::endl;

    // Create context
    CUcontext cuContext0, cuContext1;
    r = cuCtxCreate(&cuContext0, 0, cuDevice);
    std::cout << "return of cuCtxCreate: " << r << std::endl;

    // Create module from binary file
    CUmodule cuModule;
    r = cuModuleLoad(&cuModule, "minimal.ptx");
    std::cout << "return of cuModuleLoad: " << r << std::endl;

    CUfunction sum0, sum1;
    r = cuModuleGetFunction(&sum0, cuModule, "_Z3sumv");
    std::cout << "return of cuModuleGetFunction: " << r << std::endl;

    printf("sum0: %p\n", sum0);

    std::cout << "loading on context 1 " << std::endl;
    r = cuDeviceGet(&cuDevice, 1);
    std::cout << "return of cuDeviceGet: " << r << std::endl;
    r = cuCtxCreate(&cuContext1, 0, cuDevice);
    std::cout << "return of cuCtxCreate: " << r << std::endl;
    r = cuModuleLoad(&cuModule, "minimal.ptx");
    std::cout << "return of cuModuleLoad: " << r << std::endl;
    r = cuModuleGetFunction(&sum1, cuModule, "_Z3sumv");
    std::cout << "return of cuModuleGetFunction: " << r << std::endl;

    printf("sum1: %p\n", sum1);

    r = cuCtxSetCurrent(cuContext0);
    std::cout << "return of cuCtxSetCurrent 0: " << r << std::endl;

    r = cuLaunchKernel(sum0, 1, 1, 1, 4, 1, 1, 0, 0, 0, 0);
    std::cout << "return of cuLaunchKernel: " << r << std::endl;

    std::cout << "!!!  switching context to device 1!" << std::endl;

    r = cuCtxSetCurrent(cuContext1);
    std::cout << "return of cuCtxSetCurrent 1: " << r << std::endl;

    r = cuLaunchKernel(sum1, 1, 1, 1, 4, 1, 1, 0, 0, 0, 0);
    std::cout << "return of cuLaunchKernel: " << r << std::endl;


    return 0;


}
