CC = nvcc
SRCS = ./src/*.cu
INC = ./src/

EXEC = bin/k_means

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC) --cudart shared

clean:
	rm -f $(EXEC)

debug:
	$(CC) $(SRCS) $(OPTS) -g -G -O0 -I$(INC) -o $(EXEC)
