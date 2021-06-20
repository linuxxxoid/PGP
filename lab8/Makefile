CC = nvcc
CFLAGS = --std=c++11 -ccbin=mpic++ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -Werror cross-execution-space-call -lm 
SOURSES = lab8.cu
BIN = run
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)