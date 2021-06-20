#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>


#define BLOCK_COUNT 256u
#define HALF_BLOCK_COUNT 128u
#define BANKS 16
#define LOG_2_BANKS 4
// macro used for computing
// Bank-Conflict-Free Shared Memory Array Indices
#define AVOID_BANK_CONFLICTS(idx) ((idx) >> BANKS + (idx) >> (LOG_2_BANKS << 1))

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)


__global__ void Histogram(unsigned char* data, int size, int* histo)
{
	// выделяем разделяемую память, объем памяти равен количеству корзинок
	__shared__ int tmp[BLOCK_COUNT];

	// вычисляем абсолютный идентификатор 
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	// смещение
	int offset = gridDim.x * blockDim.x;

	// заполним временный массив нулем,
	// фактически заполняем 256 элементов как 0 в общей памяти
	tmp[threadIdx.x] = 0;
	__syncthreads(); // потоки ожидают выполнения заполнения нулем tmp
	
	// перебираем все элементы буфера data,
	// пока абс идентификатор не коснется значения size
	while (idx < size)
	{
		// извлекаем значение, находящееся в буфере
		// и увеличиваем счетчик в массиве разделяемой памяти
		atomicAdd(&tmp[data[idx]], 1);
		idx += offset; // увеличение абс идентиф по смещение
	}
	__syncthreads(); // ждем все потоки

	// обновляем окончательный результат в массиве histo
	int i = threadIdx.x;
	while (i < BLOCK_COUNT)
	{
		atomicAdd(&histo[i], tmp[i]);
		i += blockDim.x;
	}
}


__global__ void Scan(int* histo, int* prefixSum)
{
	__shared__ int tmp[BLOCK_COUNT];

	int threadId = threadIdx.x;
	int offset = 1;

	int aIdx = threadIdx.x;
	int bIdx = threadIdx.x + HALF_BLOCK_COUNT;

	int bankOffsetA = AVOID_BANK_CONFLICTS(aIdx);
	int bankOffsetB = AVOID_BANK_CONFLICTS(bIdx);

	// загружаем данные из гистограммы в общую память
	tmp[aIdx + bankOffsetA] = histo[aIdx];
	tmp[bIdx + bankOffsetB] = histo[bIdx];

	// строим сумму на месте вверх по дереву
	{
		int lvl = BLOCK_COUNT >> 1;

		while (lvl > 0)
		{
			__syncthreads();

			if (threadId < lvl)
			{
				int aIndex = (offset * (threadId * 2 + 1) - 1);
				int bIndex = (offset * (threadId * 2 + 2) - 1);
				aIndex += AVOID_BANK_CONFLICTS(aIndex);
				bIndex += AVOID_BANK_CONFLICTS(bIndex); 
				tmp[bIndex] += tmp[aIndex];
			}
			offset <<= 1;
			lvl >>= 1;
		}
	}

	// очищаем последний элемент
	if (threadId == 0)
	{
		tmp[BLOCK_COUNT - 1 + AVOID_BANK_CONFLICTS(BLOCK_COUNT - 1)] = 0;
	}

	// идем вниз по "дереву" и строим сканирование
	{
		int lvl = 1; 
		while (lvl < BLOCK_COUNT)
		{
			offset >>= 1;
			__syncthreads();
			if (threadId < lvl)
			{
				int aIndex = (offset * (threadId * 2 + 1) - 1);
				int bIndex = (offset * (threadId * 2 + 2) - 1);
				aIndex += AVOID_BANK_CONFLICTS(aIndex);
				bIndex += AVOID_BANK_CONFLICTS(bIndex);
				int temp = tmp[aIndex];
				tmp[aIndex] = tmp[bIndex];
				tmp[bIndex] += temp; 
			}
			lvl <<= 1;
		}
	} 

	__syncthreads();
	// записываем результаты в массив prefixSum
	prefixSum[aIdx] = histo[aIdx] + tmp[aIdx + bankOffsetA];
	prefixSum[bIdx] = histo[bIdx] + tmp[bIdx + bankOffsetB];
	
}



__global__ void CountSort(unsigned char* data, int* prefixSum, unsigned char* result, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	int i = idx, j;
	while (i < size)
	{
		// j = prefixSum[i] - 1;
		// bound = i ? prefixSum[i - 1] : 0;

		// while (j >= bound)
		// {
		// 	data[j] = i;
		// 	--j;
		// }
		j = atomicSub(&prefixSum[data[i]], 1) - 1;
		result[j] = data[i];
		i += offset;
	}
}


int main()
{
	int size;

	freopen(NULL, "rb", stdin);
	fread(&size, sizeof(int), 1, stdin);

    unsigned char* data = new unsigned char[size];
	
	fread(data, sizeof(unsigned char), size, stdin);
	fclose(stdin);

    unsigned char* deviceData;
    unsigned char* deviceResult;
    int* deviceHisto;
    int* devicePrefix;

	float elapsedTime;
	cudaEvent_t start, stop;

	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));

	CSC(cudaMalloc((void**)&deviceData, sizeof(unsigned char) * size));
	CSC(cudaMemcpy(deviceData, data, sizeof(unsigned char) * size, cudaMemcpyHostToDevice));

	CSC(cudaMalloc((void**)&deviceHisto, sizeof(int) * BLOCK_COUNT));
	CSC(cudaMalloc((void**)&devicePrefix, sizeof(int) * BLOCK_COUNT));
	CSC(cudaMemset(deviceHisto, 0, sizeof(int) * BLOCK_COUNT));

	CSC(cudaMalloc((void**)&deviceResult, sizeof(unsigned char) * size));

	CSC(cudaEventRecord(start));

	Histogram<<<BLOCK_COUNT, BLOCK_COUNT>>>(deviceData, size, deviceHisto);
    cudaThreadSynchronize(); // wait end
	CSC(cudaGetLastError());

	Scan<<<1, HALF_BLOCK_COUNT>>>(deviceHisto, devicePrefix);
    cudaThreadSynchronize(); // wait end
	CSC(cudaGetLastError());

	CountSort<<<1, BLOCK_COUNT>>>(deviceData, devicePrefix, deviceResult, size);
    cudaThreadSynchronize(); // wait end
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));

	CSC(cudaEventElapsedTime(&elapsedTime, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("Time :  %f ms \n", elapsedTime);

	CSC(cudaMemcpy(data, deviceResult, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost));

	// freopen(NULL, "wb", stdout);
	// fwrite(data, sizeof(unsigned char), size, stdout);
	// fclose(stdout);

	CSC(cudaFree(deviceData));
	CSC(cudaFree(deviceHisto));
	CSC(cudaFree(devicePrefix));

	delete[] data;
	return 0;
}