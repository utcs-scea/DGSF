.PHONY: all

CUDAFLAGS = -O3 --cudart shared

all:
	nvcc migration_bench.cu -o migration_bench ${CUDAFLAGS}
	nvcc minimal.cu -o minimal ${CUDAFLAGS}
	chmod u+x migration_bench
	chmod u+x minimal
	
clean:
	rm -f minimal migration_bench
