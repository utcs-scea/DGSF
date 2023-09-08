CC = g++
NVCC = nvcc
SRCS = ./src/common.cpp
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -O3
CUDAOPTS = -std=c++14 -Wall -Werror -O3

clean:
	rm -f bin/*

cuda-basic:
	$(NVCC) ./src/cuda-basic.cu $(SRCS) -Xcompiler "$(CUDAOPTS)" -I$(INC) -o bin/cuda-basic --cudart shared
	
cuda-shared:
	$(NVCC) ./src/cuda-shared.cu $(SRCS) -Xcompiler "$(CUDAOPTS)" -I$(INC) -o bin/cuda-shared --cudart shared
	
sequential:
	$(CC) ./src/sequential.cpp $(SRCS) $(OPTS) -I$(INC) -o bin/sequential
	
thrust:
	$(NVCC) ./src/thrust.cu $(SRCS) -Xcompiler "$(CUDAOPTS)" -I$(INC) -o bin/thrust
