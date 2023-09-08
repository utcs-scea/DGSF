CC = g++
OPTS = -std=c++17 -Wall -Werror -lpthread -O3
CUDA_OPTS = -Xcompiler -O3 --cudart shared

EXEC = bin/*
SEQ_BIN = bin/sequential

all: clean sequential cuda_float cuda_double thrust_impl

sequential:
	$(CC) src/sequential.cpp $(OPTS) -o bin/sequential
cuda_float:
	nvcc src/cuda_using_float_atomic_add.cu $(CUDA_OPTS)  -o bin/cuda_float
cuda_double:
	nvcc src/cuda_using_double_atomic_add.cu $(CUDA_OPTS)  -o bin/cuda_double
thrust_impl:
	nvcc src/thrust_impl.cu $(CUDA_OPTS)  -o bin/thrust_impl
clean:
	rm -f $(EXEC)
