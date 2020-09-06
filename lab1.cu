#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// blockDim.x, y, z gives the number of threads in a block, in the particular direction
// gridDim.x, y, z gives the number of blocks in a grid, in the particular direction
// blockDim.x * gridDim.x gives the number of threads in a grid(in the x direction, in this case)


// blockIdx.x идентификатор текущего блока
// threadIdx.x это идентификатор текущего идентификатора потока.
// поскольку gridDim.x указывает количество блоков в сетке в направлении X,
// а blockDim.x сообщает, что количество потоков в блоке.
// поэтому(blockDim.x* gridDim.x) дает нам общее количество потоков в grid
// __global__ void Reverse(double* vec, double* res, int size) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    //  на общее количество потоков, работающих в сетке
//    int offset = blockDim.x * gridDim.x;
//    while (idx < size)
//    {
//        res[idx] = vec[size - 1 - idx];
//        idx += offset;
//    }
//}
__global__ void reverseArrayBlock(double* res, double* vec, int size)
{



   // GridDim.x gives no. of block in grid in X dimention

 /*  if (new_id < size) {
       d_b[old_id] = d_a[new_id];

   }
   else {
       d_b[old_id] = d_a[old_id];
   }*/
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // на общее количество потоков, работающих в сетке
   int offset = gridDim.x * blockDim.x - 1 - idx;
   res[idx] = vec[offset];

}

int main(int argc, const char* argv[]) {
   int size = 5;
   //std::cin >> size;
   if (size < 0) {
       std::cerr << "Too small size'\n";
       return 0;
   }

   int accuracy = 10;
   double* hostVec = new double[size];

   for (int i = 0; i < size; ++i) {
       //std::cin >> hostVec[i];
       hostVec[i] = i;
       std::cout << hostVec[i] << '\t';
   }

   double* deviceVec, * deviceRes;

   // Выделяем память для device копий
   cudaMalloc((void**)&deviceVec, sizeof(double) * size);
   cudaMalloc((void**)&deviceRes, sizeof(double) * size);
   // Копируем ввод на device
   cudaMemcpy(deviceVec, hostVec, sizeof(double) * size, cudaMemcpyHostToDevice);

   int nBlockCount = 1;
   int nThreadCount = size;
   // Now launch your kernel using the appropriate macro:
   reverseArrayBlock <<<nBlockCount, nThreadCount>>>(deviceRes, deviceVec, size);

   checkCUDAError("kernel invocation");
   cudaMemcpy(hostVec, deviceRes, sizeof(double) * size, cudaMemcpyDeviceToHost);
   checkCUDAError("memcpy");

   std::cout.precision(accuracy);
   std::cout.setf(std::ios::scientific);
   for (int i = 0; i < size; ++i) {
       std::cout << hostVec[i] << ' ';
   }
   std::cout << std::endl;

   cudaFree(deviceVec);
   cudaFree(deviceRes);
   delete[] hostVec;

   return 0;
}
