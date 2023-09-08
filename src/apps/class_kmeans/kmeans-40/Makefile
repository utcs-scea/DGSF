CC = g++ 
NVCC = nvcc 
SRCS = ./src/*.cpp
SRCS_CU = ./src/*.cu
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -O3
OPTS_NVCC = -O3 --cudart shared #-std=c++14 

EXEC_SEQ = bin/kmeans_seq
EXEC_THRUST = bin/kmeans_thrust
EXEC_CUDA = bin/kmeans_cuda
EXEC_CUDA_SHMEM = bin/kmeans_cuda_shmem

all: clean compile_seq compile_thrust compile_cuda compile_cuda_shmem

compile_seq:
	$(CC) $(SRCS) -DKMEANS_SEQ $(OPTS) -I$(INC) -o $(EXEC_SEQ)

compile_thrust:
	$(NVCC) -c src/main.cpp -DKMEANS_THRUST $(OPTS_NVCC) -I$(INC) -o bin/main_thrust.o
	$(NVCC) -c src/io.cpp -DKMEANS_THRUST $(OPTS_NVCC) -I$(INC) -o bin/io_thrust.o
	$(NVCC) -c src/argparse.cpp -DKMEANS_THRUST $(OPTS_NVCC) -I$(INC) -o bin/argparse_thrust.o
	$(NVCC) -c src/kmeans_thrust.cu -DKMEANS_THRUST $(OPTS_NVCC) -I$(INC) -o bin/kmeans_thrust.o
	$(NVCC) bin/main_thrust.o bin/io_thrust.o bin/argparse_thrust.o bin/kmeans_thrust.o $(OPTS_NVCC) -o $(EXEC_THRUST)

compile_cuda:
	$(NVCC) -c src/main.cpp -DKMEANS_CUDA $(OPTS_NVCC) -I$(INC) -o bin/main_cuda.o
	$(NVCC) -c src/io.cpp -DKMEANS_CUDA $(OPTS_NVCC) -I$(INC) -o bin/io_cuda.o
	$(NVCC) -c src/argparse.cpp -DKMEANS_CUDA $(OPTS_NVCC) -I$(INC) -o bin/argparse_cuda.o
	$(NVCC) -c src/kmeans_cuda.cu -DKMEANS_CUDA $(OPTS_NVCC) -I$(INC) -o bin/kmeans_cuda.o
	$(NVCC) bin/main_cuda.o bin/io_cuda.o bin/argparse_cuda.o bin/kmeans_cuda.o $(OPTS_NVCC) -o $(EXEC_CUDA)

compile_cuda_shmem:
	$(NVCC) -c src/main.cpp -DKMEANS_CUDA_SHMEM $(OPTS_NVCC) -I$(INC) -o bin/main_cuda_shmem.o
	$(NVCC) -c src/io.cpp -DKMEANS_CUDA_SHMEM $(OPTS_NVCC) -I$(INC) -o bin/io_cuda_shmem.o
	$(NVCC) -c src/argparse.cpp -DKMEANS_CUDA_SHMEM $(OPTS_NVCC) -I$(INC) -o bin/argparse_cuda_shmem.o
	$(NVCC) -c src/kmeans_cuda_shmem.cu -DKMEANS_CUDA_SHMEM $(OPTS_NVCC) -I$(INC) -o bin/kmeans_cuda_shmem.o
	$(NVCC) bin/main_cuda_shmem.o bin/io_cuda_shmem.o bin/argparse_cuda_shmem.o bin/kmeans_cuda_shmem.o $(OPTS_NVCC) -o $(EXEC_CUDA_SHMEM)

clean:
	rm -f bin/*.o $(EXEC_SEQ) $(EXEC_THRUST) $(EXEC_CUDA) $(EXEC_CUDA_SHMEM)
