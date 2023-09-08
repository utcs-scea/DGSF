CC = g++ 
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -lpthread -O3

EXEC = bin/kmeans_seq

all: seqAll thrustAll cudaAll cudaShmemAll

seqAll: cleanSeq seq

# all: clean seq

# seq:
# 	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)
seq:
	g++ ./src/kmeans_seq.cpp ./src/io.cpp ./src/argparse.cpp -std=c++17 -Wall -Werror -lpthread -O3 -I./src/ -o bin/kmeans_seq
	
# g++  ./src/*.cpp -std=c++17 -Wall -Werror -lpthread -O3 -I./src/ -o bin/kmeans_seq

cleanSeq:
	rm -f $(EXEC)

thrust:
	nvcc ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o bin/kmeans_thrust
	
# nvcc ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o thrust_test
# nvcc -Xcompiler "-std=c++17" ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o thrust_test
# nvcc -Xcompiler "$(OPTS)" ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o thrust_test
# nvcc ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o thrust_test
# nvcc ./src/kmeans_thrust.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_thrust.cu -o bin/kmeans_thrust

cleanThrust:
	rm -f bin/kmeans_thrust

# rm -f thrust_test
	
thrustAll: cleanThrust thrust

cuda:
	nvcc ./src/kmeans_cuda.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda.cu -o bin/kmeans_cuda
	
# nvcc ./src/kmeans_cuda.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda.cu -o bin/kmeans_cuda
# nvcc ./src/kmeans_cuda.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda.cu -o cuda_test

cleanCuda:
	rm -f bin/kmeans_cuda

# rm -f cuda_test
	
cudaAll: cleanCuda cuda

cudaShmem:
	nvcc ./src/kmeans_cuda_shmem.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda_shmem.cu -o bin/kmeans_cuda_shmem
	
# nvcc ./src/kmeans_cuda_shmem.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda_shmem.cu -o bin/kmeans_cuda_shmem
# nvcc ./src/kmeans_cuda_shmem.cpp ./src/io.cpp ./src/argparse.cpp ./src/kmeans_cuda_shmem.cu -o cuda_shmem_test

cleanCudaShemem:
	rm -f bin/kmeans_cuda_shmem

# rm -f cuda_shmem_test
	
cudaShmemAll: cleanCudaShemem cudaShmem