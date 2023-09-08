CC = g++
CC_OPTS = -std=c++17 -Wall -Werror -O3

NVCC = nvcc
NVCC_OPTS = -O3  --cudart shared

all: bin clean sequential thrust cuda_basic cuda_shmem

bin:
	mkdir -p bin

sequential: bin
	$(CC) ./src_sequential/*.cpp $(CC_OPTS) -I./src_sequential -o ./bin/kmeans_sequential

thrust: bin
	$(NVCC) ./src_thrust/*.cpp ./src_thrust/*.cu $(NVCC_OPTS) -I./src_thrust -o ./bin/kmeans_thrust

cuda_basic: bin
	$(NVCC) ./src_cuda_basic/*.cpp ./src_cuda_basic/*.cu $(NVCC_OPTS) -I./src_cuda_basic -o ./bin/kmeans_cuda_basic

cuda_shmem: bin
	$(NVCC) ./src_cuda_shmem/*.cpp ./src_cuda_shmem/*.cu $(NVCC_OPTS) -I./src_cuda_shmem -o ./bin/kmeans_cuda_shmem

clean: bin
	rm -f ./bin/kmeans_*

