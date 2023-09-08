CC = g++ 
NVCC = nvcc
SRCS = ./src/*.cpp
INC = ./src/
STD = -std=c++14
OPTS = $(STD) -Wall -Werror -O3
NOPTS = $(STD) -O3 --compiler-options=-Wall --compiler-options=-Werror --cudart shared

BIN = bin
EXEC = $(BIN)/kmeans
EXEC_CUDA = $(BIN)/kmeans_cuda
EXEC_THRUST = $(BIN)/kmeans_thrust

clean:
	rm -f $(BIN)/*

compile:
	$(CC) main.cpp $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
	
kmeans_kernel.o:
	$(NVCC) -arch=sm_60 -c src-cuda/kmeans_kernel.cu -o $(BIN)/kmeans_kernel.o

cuda-compile: kmeans_kernel.o
	$(NVCC) main_cuda.cu $(SRCS) $(BIN)/kmeans_kernel.o $(NOPTS) -I $(INC) -I ./src-cuda/ -o $(EXEC_CUDA)
	rm -f $(BIN)/kmeans_kernel.o

thrust-compile:
	$(NVCC) main_thrust.cu $(SRCS) $(NOPTS) -I $(INC) -o $(EXEC_THRUST)
