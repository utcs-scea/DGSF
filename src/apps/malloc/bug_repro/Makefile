all:
	nvcc --ptx minimal.cu
	nvcc test.cu -lcuda -o test
	nvcc mix.cu -lcuda -o mix
	nvcc working.cu -lcuda -o working