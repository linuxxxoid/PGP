#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)



struct comparator {
    __host__ __device__ double fabs(double a){
        return  a < 0.0 ? -a : a;
    }

    __host__ __device__ bool operator()(double a, double b)
    {
        return fabs(a) < fabs(b);
    }
};



__host__ void Printer(double* matrix, int height, int width)
{
    std::cout << "Printer\n";
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
          printf("a[i=%d, j=%d->%d] = %.1f ", i, j, j * width + i, matrix[j * width + i]);
        }
        printf("\n");
    }

}


__global__ void SwapGPU(double* matrix, int width, int height, int row, int rowWithMax)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int xOffset = gridDim.x * blockDim.x;
    double tmp;

    for (int i = idx + row; i < height; i += xOffset)
    {
        tmp = matrix[i * width + row];
        matrix[i * width + row] = matrix[i * width + rowWithMax];
        matrix[i * width + rowWithMax] = tmp;
    }
}


__global__ void Normalization(double* matrix, int width, int height, int row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int xOffset = gridDim.x * blockDim.x;

    for (int i = idx + row + 1; i < height; i += xOffset)
    {
        matrix[i * width + row] /= matrix[row * width + row]; 
    }
}


__global__ void ForwardGauss(double* matrix, int width, int height, int row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int xOffset = gridDim.x * blockDim.x;
    int yOffset = gridDim.y * blockDim.y;

    for (int i = idx + row + 1; i < width; i += xOffset)
    {
        for (int j = idy + row + 1; j < height; j += yOffset)
        {
            matrix[j * width + i] -= matrix[j * width + row] * matrix[row * width + i];
        }
    }
}


__global__ void BackwardGauss(double* matrix, double* x, int size, int row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int xOffset = gridDim.x * blockDim.x;

    for (int i = row - 1 - idx; i >= 0; i -= xOffset)
    {
        x[i] -= matrix[row * size + i] * x[row];
    }
}



int main(int argc, const char* argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int size;
    std::cin >> size;
    int height = size + 1;
    int width = size;
    double* matrix = new double[height * width];

      for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cin >> matrix[j * width + i];
        }
    }

    for (int i = 0; i < size; ++i)
    {
        std::cin >> matrix[size * size + i];
    }

    double* matrixGPU;
    CSC(cudaMalloc(&matrixGPU, sizeof(double) * height * width));
    CSC(cudaMemcpy(matrixGPU, matrix, sizeof(double) * height * width, cudaMemcpyHostToDevice));
    
    int xThreadCount = 32;
    int yThreadCount = 32;

    int xBlockCount = 32;
    int yBlockCount = 32;

    comparator comp;
    thrust::device_ptr<double> ptr, ptrMax;
    int rowWithMax;
    for (int row = 0; row < size - 1; ++row)
    {
        ptr = thrust::device_pointer_cast(matrixGPU + row * size);
        ptrMax = thrust::max_element(ptr + row, ptr + size, comp);
        rowWithMax = ptrMax - ptr;

        if (rowWithMax != row)
        {
            SwapGPU<<<dim3(xBlockCount * yBlockCount), dim3(xThreadCount * yThreadCount)>>>(matrixGPU, width, height, row, rowWithMax);
            CSC(cudaGetLastError());
        }
        Normalization<<<dim3(xBlockCount * yBlockCount), dim3(xThreadCount * yThreadCount)>>>(matrixGPU, width, height, row);
        CSC(cudaGetLastError());

        ForwardGauss<<<dim3(xBlockCount, yBlockCount), dim3(xThreadCount, yThreadCount)>>>(matrixGPU, width, height, row);
        CSC(cudaGetLastError());    
    }
    CSC(cudaMemcpy(matrix, matrixGPU, sizeof(double) * width * height, cudaMemcpyDeviceToHost));

    double* x = new double[size];

    for (int i = 0; i < size; ++i)
    {
        x[i] = matrix[width * width + i];
    }
    x[size - 1] /= matrix[(width - 1) * width + (width - 1)];

    double* xGPU;
    CSC(cudaMalloc(&xGPU, sizeof(double) * size));
    CSC(cudaMemcpy(xGPU, x, sizeof(double) * size, cudaMemcpyHostToDevice));

    for (int row = size - 1; row > 0; --row)
    {
        BackwardGauss<<<dim3(xBlockCount * yBlockCount), dim3(xThreadCount * yThreadCount)>>>(matrixGPU, xGPU, size, row);
        CSC(cudaGetLastError());
    }
    
    CSC(cudaMemcpy(x, xGPU, sizeof(double) * size, cudaMemcpyDeviceToHost));

    const int accuracy = 10;

    for (int i = 0; i < size - 1; ++i)
    {
        std::cout << std::scientific << std::setprecision(accuracy) << x[i] << " ";
    }
    std::cout << std::scientific << std::setprecision(accuracy) << x[size - 1];

    CSC(cudaFree(matrixGPU));
    CSC(cudaFree(xGPU));

    delete[] matrix;
    delete[] x;

    return 0;
}