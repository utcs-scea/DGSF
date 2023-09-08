# Target rules
all:
	nvcc -o kmeans.o -c kmeans.cpp
	nvcc -o kmeans_kernel.o -c kmeans_kernel.cu --cudart shared
	nvcc -o kmeans kmeans.o kmeans_kernel.o --cudart shared

clean:
	rm -f kmeans.o kmeans kmeans_kernel.o

