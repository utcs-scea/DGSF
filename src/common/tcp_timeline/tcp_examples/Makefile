.PHONY: all clean

CPPFLAGS = -O3 -std=c++11 -g
CC = g++
INCLUDE_DIR = ../
DEPS = ../tcp_bench.cpp

EXS = client_ex server_ex multiclient_ex

all: ${EXS}

${EXS}: ${DEPS}
	${CC} -o $@ $@.cpp ${DEPS} ${CPPFLAGS} -I ${INCLUDE_DIR}

clean:
	rm ${EXS}
