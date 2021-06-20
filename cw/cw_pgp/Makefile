CC = nvcc
CFLAGS = -ccbin=mpic++ --std=c++11 -Xcompiler -fopenmp
FILES = cw.cu
PROG = run

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)