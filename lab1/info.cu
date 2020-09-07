#include <stdio.h>

int main()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  totalGlobalMem: %lu\n", prop.totalGlobalMem);
      printf(" Const Mem : %lu\n", prop.totalConstMem);
      printf("Max shared mem for blocks %lu\n", prop.sharedMemPerBlock);
      printf("max regs per block %d\n", prop.regsPerBlock);
      printf("Max thread per block %d\n", prop.maxThreadsPerBlock);
      printf("multiProcessorCount : %d\n", prop.multiProcessorCount);
      printf("maxThreadsDim %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("maxGridSize %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
