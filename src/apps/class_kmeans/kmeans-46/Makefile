NVCC=nvcc
NVCCFLAGS=-O3 -m64 --cudart shared
CC = g++ 
EXEC = exec
INC = ./src/*.h
COMMON_FILES      :=  src/kmeans.cpp src/kmeans_init.cpp src/io.cpp src/argparse.cpp src/kmeans_seq.cpp
SEQ_FILES         :=  src/kmeans_seq.cpp
THRUST_FILES      :=  src/kmeans_thrust.cu
CUDA_FILES        :=  src/kmeans_cuda.cu
OPTS               = -std=c++17 -Wall -Werror


kmeans_seq:
	$(CC) -DKMEANS_SEQ $(COMMON_FILES) $(OPTS) -I$(INC) -o $(EXEC)/kmeans_seq

kmeans_thrust:
	$(NVCC) $(NVCCFLAGS) -DKMEANS_THRUST $(COMMON_FILES) $(THRUST_FILES) -I$(INC) -o $(EXEC)/kmeans_thrust

kmeans_cuda_basic: 
	$(NVCC) $(NVCCFLAGS) -DKMEANS_CUDA_BASIC $(COMMON_FILES) $(CUDA_FILES) -I$(INC) -o $(EXEC)/kmeans_cuda_basic

kmeans_cuda_shmem: 
	$(NVCC) $(NVCCFLAGS) -DKMEANS_CUDA_SHMEM $(COMMON_FILES) $(CUDA_FILES) -I$(INC) -o $(EXEC)/kmeans_cuda_shmem

clean:
	rm -f $(EXEC)/*
