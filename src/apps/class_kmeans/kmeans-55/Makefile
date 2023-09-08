CC = nvcc
SRCS = ./src/*.cpp ./src/*.cu
INC = ./src/
OPTS = --std=c++11 --cudart shared

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) -O3 $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

debug:
	$(CC) $(SRCS) $(OPTS) -DDEBUG -I$(INC) -o $(EXEC) -g

timing:
	$(CC) $(SRCS) $(OPTS) -DTIMING -I$(INC) -o $(EXEC) -g

clean:
	rm -f $(EXEC)
