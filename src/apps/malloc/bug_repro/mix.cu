#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>



int main() {
    CUresult r;


    cudaSetDevice(0);
    cudaFree(0);

    cudaSetDevice(1);
    cudaFree(0);

    cudaSetDevice(0);

    //cuInit(0);

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

    cudaSetDevice(1);

    r = cuModuleLoad(&cuModule, "minimal.ptx");
    std::cout << "return of cuModuleLoad: " << r << std::endl;

    r = cuModuleGetFunction(&sum, cuModule, "_Z3sumv");
    std::cout << "return of cuModuleGetFunction: " << r << std::endl;

    r = cuLaunchKernel(sum, 1, 1, 1, 4, 1, 1, 0, 0, 0, 0);
    std::cout << "return of cuLaunchKernel: " << r << std::endl;


    return 0;


}
