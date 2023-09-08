#include "kmeans.h"
#include <stdio.h>

__global__ void mykernel() {
}

void kernel_launch() {
	printf("Launching it");
	mykernel<<<1,1>>>();
}
