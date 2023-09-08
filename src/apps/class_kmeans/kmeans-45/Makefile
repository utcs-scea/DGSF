CC=nvcc
CFLAGS=-I. --cudart shared
BUILDDIR=build/
DEPS = lib/datasets.h

sequential: sequential.o lib/argparse.o lib/datasets.o
	$(CC) -o sequential sequential.o lib/argparse.o lib/datasets.o

thrust.o: thrust.cu
	$(CC) -c -o thrust.o thrust.cu

thrust: thrust.o  lib/argparse.o lib/datasets.o
	$(CC) -o thrust thrust.o  lib/argparse.o lib/datasets.o

kmeans_kernels.o: kmeans_kernels.cu
	$(CC) -c -o kmeans_kernels.o kmeans_kernels.cu -arch=sm_60 --cudart shared

cuda.o: cuda.cpp
	$(CC) -c -o cuda.o cuda.cpp --cudart shared

cuda: cuda.o  lib/argparse.o lib/datasets.o kmeans_kernels.o
	$(CC) -o cuda cuda.o  lib/argparse.o lib/datasets.o kmeans_kernels.o  --cudart shared

cudashared.o: cudashared.cpp
	$(CC) -c -o cudashared.o cudashared.cpp --cudart shared

cudashared: cudashared.o  lib/argparse.o lib/datasets.o kmeans_kernels.o
	$(CC) -o cudashared cudashared.o  lib/argparse.o lib/datasets.o kmeans_kernels.o --cudart shared
