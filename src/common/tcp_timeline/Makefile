.PHONY: all clean

GCC = g++
GCCFLAGS = -Wall -g
CPPFILES = ${wildcard *.cpp}
FILES = ${basename ${CPPFILES}}
OFILES = ${addsuffix .o, ${FILES}}

all: clean server

server: 
	${GCC} server.cpp tcp_bench.cpp ${GCCFLAGS} -o $@

clean:
	rm -f *.o server

