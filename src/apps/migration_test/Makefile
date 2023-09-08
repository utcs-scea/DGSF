ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >
SHELL := $(shell which bash)
.ONESHELL:
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

CCFLAGS = -Wall -O3 -g #-DTEST
NVCCFLAGS = -O3 -g --ptxas-options=-v --cudart shared -pg #-DTEST #-DDEBUG
EXNVCCFLAGS = -O3 -g --ptxas-options=-v --cudart shared -pg #-DTEST #-DDEBUG

THIS_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
WORK_DIR = ${THIS_DIR}/../..
INCLUDE_DIR = ${WORK_DIR}/common
 
EXEC = test
CCFILES = ${wildcard *.cc}
CUFILES = ${wildcard *.cu}
FILES = ${basename ${CCFILES}} ${basename ${CUFILES}}
OFILES = ${addsuffix .o,${FILES}}

IN_FILES=${sort ${wildcard *.in}}
TESTS=${subst .in,,${IN_FILES}}
TEST_OUTS=${addsuffix .out,${TESTS}}

.PHONY: all clean indocker

${EXEC}: ${OFILES}
> nvcc -o $@ ${EXNVCCFLAGS} $^ -I ${INCLUDE_DIR}

indocker:
> docker run --rm -it -v ${WORK_DIR}:${WORK_DIR} -u1031:502 --gpus all -w ${PWD} nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04 make -C ${THIS_DIR}

%.o: %.cc Makefile
> nvcc -o $*.o ${NVCCFLAGS} -c $<

%.o: %.cu Makefile
> nvcc -o $*.o ${NVCCFLAGS} -c $< -I ${INCLUDE_DIR} 
