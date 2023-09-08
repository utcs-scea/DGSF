#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "tcp_timeline/tcp_timeline_client.hpp"


#define CUDA_ERROR_CHECK
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}


#define MAX_THREADS 512
#define BLK(n) ceil((float)n / MAX_THREADS)
#define THREAD(n) min(n, MAX_THREADS)

#define PORT 50057

__global__ void sum(float* dest, float* src, size_t n, size_t loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        for (int i = 0 ; i < loops ; i++) {
            float x1 = dest[tid];
            float y1 = src[tid];
            float x2 = y1+1;
            float y2 = x1+2;
            float dx = x1-x2;
            float dy = y1-y2;
            float dist = sqrtf(dx*dx + dy*dy);
            if (i%2)
                dest[tid] = dest[tid]+dist+0.5;
            else
                dest[tid] = dest[tid]-dist-0.5;
        }
    }
}

void readInputs(float* a, std::ifstream& input, size_t n) {
    std::string line;
    std::getline(input, line, '\n');
    std::istringstream iss(line);
    std::string elem;
    for (size_t i = 0; i < n; i++) {
        getline(iss, elem, ',');
        a[i] = std::stof(elem);
    }
}

int main(int argc, char* argv[]) {
    size_t n = 0;
    std::string infile;
    size_t nkernels = 1;
    size_t nloops = 10000;

    for (int i = 1; i < argc; i+=2) {
        if (strcmp("-n", argv[i]) == 0) {
            n = std::stoul(argv[i+1]);
        }
        else if (strcmp("-f", argv[i]) == 0) {
            infile = argv[i+1];
        }
        else if (strcmp("-k", argv[i]) == 0) {
            nkernels = std::stoul(argv[i+1]);
        }
        else if (strcmp("-l", argv[i]) == 0) {
            nloops = std::stoul(argv[i+1]);
        }
    }

    TCPTimelineClient logger;
    logger.notify(START);

#ifdef DEBUG
    std::cout << "n: " << n << "\n";
    std::cout << "infile: " <<  infile << "\n";
#endif

    float* h_a = (float*)malloc(sizeof(float) * n);
    float* h_b = (float*)malloc(sizeof(float) * n);
    float *d_a, *d_b;
    
    if (infile.empty()) {
        srand(0);
        for (int i = 0 ; i < n ; i++) {
            h_a[i] = float(rand())/float((RAND_MAX)) * 5.0;
            h_b[i] = float(rand())/float((RAND_MAX)) * 5.0;
        }
    } else {
        std::ifstream input(infile);
        readInputs(h_a, input, n);
        readInputs(h_b, input, n);
    }

    std::cout << "before [0]: " << h_a[0] << std::endl;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    CudaCheckError();
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    CudaCheckError();

    logger.notify(KERNEL);

    // sum vectors into d_a
    for (int i = 0 ; i < nkernels ; i++) {
        sum<<<BLK(n), THREAD((int)n)>>>(d_a, d_b, n, nloops);
        CudaCheckError();
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    //float acc = 0;
    //for (int i = 0 ; i < n ; i++)
    //    acc += h_a[i];
    //std::cout << "result: " << acc << std::endl;

    std::cout << "after [0]: " << h_a[0] << std::endl;

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    logger.notify(END);

    return 0;
}
