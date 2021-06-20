CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -rdc=true
FILES = cw.cu
PROG = run

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)