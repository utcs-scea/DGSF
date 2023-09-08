.PHONY: all indocker run-docker clean

NVCC = nvcc
CUDAFLAGS = -O3 --cudart shared

EXEC = vectorsum
CUFILES = ${wildcard *.cu}
FILES = ${basename ${CUFILES}}

THIS_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
WORK_DIR = ${THIS_DIR}/../..
INCLUDE_DIR = ${WORK_DIR}/common
OFILES = ${addsuffix .o, ${FILES}} 

all: ${EXEC}

${EXEC}: ${OFILES}
	${NVCC} -o ${EXEC} ${CUDAFLAGS} $^ 
	chmod u+x ${EXEC}

%.o: %.cu
	${NVCC} -o $*.o ${CUDAFLAGS} -c $< -I ${INCLUDE_DIR}

indocker: Makefile
	docker run --rm -it -v ${WORK_DIR}:${WORK_DIR} -u1031:502 --gpus all -w ${PWD} nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04 make -C ${THIS_DIR} ${EXEC}

run-docker: Makefile
	docker run --rm -it -v ${WORK_DIR}:${WORK_DIR} -u1031:502 --gpus all -w ${PWD} nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04

clean:
	rm -f ${EXEC} *.o  ${TCP}
