NVCCFLAGS := -cubin -arch sm_70 -O3 
CFLAGS := -I$(shell python3 -c 'import cupy; print(cupy.__path__[0])')/_core/include

.PHONY: all
all:
	nvcc -o zoom_kernel.cubin $(NVCCFLAGS) $(CFLAGS) zoom_kernel.cu

clean:
	rm -f zoom_kernel.cubin
