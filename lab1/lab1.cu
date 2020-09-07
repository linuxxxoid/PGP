#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>


void checkCudaError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(0);
    }
}


__global__ void Reverse(double* res, double* vec)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x - 1 - idx;
    res[idx] = vec[offset];
}


int main(int argc, const char* argv[])
{
    int size;
    std::cin >> size;
    //scanf("%lli", &size); 
    std::cout << "size=" << size;
    const int MAX = 33554432;
    const int MIN = 0;
    if (size < MIN && size > MAX)
    {
       std::cerr << "ERROR: Incorrect size!\n";
       exit(0);
    }

    double* hostVec = new double[size];

    for (int i = 0; i < size; ++i)
    {
        //scanf("lf", &hostVec[i]);
        std::cin >> hostVec[i];
        //std::cout << hostVec[i] << '\t';
    }
    std::cout << "WHAT WE HAVE\n";
    for (int i = 0; i < size; ++i)
    {
       std::cout << hostVec[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "WHAT WE HAVE\n";

    double *deviceVec, *deviceRes;

    // Выделяем память для device копий
    cudaMalloc((void**) &deviceVec, sizeof(double) * size);
    cudaMalloc((void**) &deviceRes, sizeof(double) * size);
    // Копируем ввод на device
    cudaMemcpy(deviceVec, hostVec, sizeof(double) * size, cudaMemcpyHostToDevice);

    int nBlockCount = 1;
    int nThreadCount = size;
    // Запускаем kernel
    Reverse<<<nBlockCount, nThreadCount>>>(deviceRes, deviceVec);

    checkCudaError("Kernel invocation");
    cudaMemcpy(hostVec, deviceRes, sizeof(double) * size, cudaMemcpyDeviceToHost);
    checkCudaError("Memcpy");

    int accuracy = 10;
    std::cout.precision(accuracy);
    std::cout.setf(std::ios::scientific);
    for (int i = 0; i < size; ++i)
    {
        std::cout << std::fixed << std::scientific;
        std::cout << std::setprecision(accuracy) << hostVec[i];

        if (i < size - 1)
            std::cout << " ";
        else
            std::cout << std::endl;
    }
    cudaFree(deviceVec);
    cudaFree(deviceRes);
    delete[] hostVec;

    return 0;
}
