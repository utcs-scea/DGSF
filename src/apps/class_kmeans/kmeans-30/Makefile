NVCC = nvcc
SRCS = ./src/*.cpp ./src/*.cu
INC = ./src/
OPTS = --cudart shared


INFILE1 = ./input/random-n2048-d16-c16.txt
INFILE2 = ./input/random-n16384-d24-c16.txt
INFILE3 = ./input/random-n65536-d32-c16.txt


EXEC = ./bin/kmean


all: clean compile 

compile:
	$(NVCC) $(SRCS) $(OPTS) -I$(INC)  -o $(EXEC) 

clean:
	rm -f $(EXEC)
	
run1: 
	$(EXEC) -i $(INFILE1) -k 16 -d 16 -m 150 -t 5e-6  -s 8675309 -r 1

run2: 
	$(EXEC) -i $(INFILE1) -k 16 -d 16 -m 150 -t 5e-6  -s 8675309 -r 2
	
run3: 
	$(EXEC) -i $(INFILE1) -k 16 -d 16 -m 150 -t 5e-6  -s 8675309 -r 3
	
run4: 
	$(EXEC) -i $(INFILE1) -k 16 -d 16 -m 150 -t 5e-6  -s 8675309 -r 4
