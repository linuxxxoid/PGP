#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "mpi.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>



#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR: Cuda in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


#define Update_ij(i, j, step, side) \
{ \
    i += step; \
    while(i >= side) \
    { \
        i -= side; \
        ++j; \
    } \
} \


const int axes = 3;
const int directions = 6;
const int BlockCount = 32;
const int ThreadsCount = 32;


void WaitAll(int* coords, int* gridProc, MPI_Request *arrOfRequests)
{
	MPI_Status tmp;

	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		if (coords[i] > 0)
		{
			MPI_Wait(&arrOfRequests[i * 2], &tmp);
		}
		if (coords[i] < gridProc[i] - 1)
		{
			MPI_Wait(&arrOfRequests[j], &tmp);
		}
	}
}


__host__ __device__
double FindMax(double a, double b, double max)
{
	double fbs = a - b;
	fbs = fbs > 0.0 ? fbs : -fbs;
	if (fbs > max)
	{
		max = fbs;
	}
	return max;
}


__host__ __device__
int GetPos(int i, int j, int k, int nY, int nX)
{
	return i + (j + k * nY) * nX;
}


void GetCoords(int* coords, int rank, int* gridProc)
{
	coords[2] = rank / gridProc[0] / gridProc[1];
	coords[1] = (rank - coords[2] * gridProc[0] * gridProc[1]) / gridProc[0];
	coords[0] = rank - (coords[2] * gridProc[1] + coords[1]) * gridProc[0];
}


int GetRank(int* coords, int* gridProc)
{
	return coords[0] + gridProc[0] * (coords[2] * gridProc[1] + coords[1]);
}


void Printer(FILE* out, double* arr, int size)
{
	for (int i = 0; i < size; ++i)
	{
		fprintf(out, "%.6e ", arr[i]);
	}
	fprintf(out, "\n");
}

void WriteOutWithMPI(int* gridProc, int* block,
					 std::string& output, double* grid, int* coords)
{
    //convert double data to char values
    int size = block[0] * block[1] * block[2];
    const int doubleSize = 14;
    char* charValues = new char[size * doubleSize];

	//memset(charValues, ' ', size * doubleSize);
    for (int k = 1; k <= block[2]; ++k)
    {
        for (int j = 1; j <= block[1]; ++j)
        {
            int i, len;
            for (i = 1; i < block[0]; ++i)
            {
            	//Возвращает количество символов, занесенных в массив.
                len = sprintf(&charValues[GetPos(i - 1, j - 1, k - 1, block[1], block[0]) * doubleSize], "%.6e ", grid[GetPos(i, j, k, block[1] + 2, block[0] + 2)]);
                if (len < doubleSize)
                {
                    charValues[GetPos(i - 1, j - 1, k - 1, block[1], block[0]) * doubleSize + len] = ' ';
                }
            }

            len = sprintf(&charValues[GetPos(i - 1, j - 1, k - 1, block[1], block[0]) * doubleSize], "%.6e\n", grid[GetPos(i, j, k, block[1] + 2, block[0] + 2)]);
            if (len < doubleSize)
            {
                charValues[GetPos(i - 1, j - 1, k - 1, block[1], block[0]) * doubleSize + len] = '\n';
            }
        }
    }
    //grid сетка процессов
	//block размер блока, который будет обрабатываться одним процессом
  	MPI_Aint stride = block[0] * doubleSize * gridProc[0];
  	MPI_Aint gstride = stride * block[1] * gridProc[1];
	MPI_Aint position = coords[0] * doubleSize * block[0];
	position += stride * block[1] * coords[1];
	position += gstride * block[2] * coords[2];

    int blocklength = block[0] * doubleSize;

	MPI_File fp;
	MPI_Datatype square, rectangle;
	//creates a vector (strided) data type with offset in bytes
	//count
	//Number of blocks (nonnegative integer).
	//blocklength
	// Number of elements in each block (nonnegative integer).
	// stride
	// Number of bytes between start of each block (integer).
	// oldtype
	// Old data type (handle).
	//                     	count       bl         stride   oldtype  newtype
	MPI_Type_create_hvector(block[1], blocklength, stride, MPI_CHAR, &square);
	MPI_Type_commit(&square);
	//                     	count    bl  stride   oldtype newtype
	MPI_Type_create_hvector(block[2], 1, gstride, square, &rectangle);
	MPI_Type_commit(&rectangle);

    MPI_File_delete(output.c_str(), MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fp);

	// fh	дескриптор файла (дескриптор)
	// disp	смещение (целое)
	// etype	элементарный тип данных (дескриптор)
	// filetype	тип файла (дескриптор)
	// datarep	представление данных (строка)
	// info	информационный объект (дескриптор)
	//disp может быть использован, для того чтобы пропустить заголовки
	// или когда файл включает последовательность сегментов данных,
	// к которым следует осуществлять доступ по различным шаблонам
	//MPI_File_set_view назначает области файла для разделения
    //процессов
	MPI_File_set_view(fp, position, MPI_CHAR, rectangle, "native", MPI_INFO_NULL);

	// 	fh
	// File handle (handle).
	// buf
	// Initial address of buffer (choice).
	// count
	// Number of elements in buffer (integer).
	// datatype
	// Data type of each buffer element (handle).
	//Collective write using individual file pointer
	MPI_File_write_all(fp, charValues, doubleSize*size, MPI_CHAR, MPI_STATUS_IGNORE);

	MPI_File_close(&fp);

	delete[] charValues;
}


void FillBuffer(double* buf, int size, double val)
{
	for (int i = 0; i < size; ++i)
	{
		buf[i] = val;
	}
}


void InitBufsEdge(double** sendBuf, double** getBuf, double** deviceSendBuf, double** deviceGetBuf, int* sizeEdges, int* gridProc, int* coords, double* u, double u0)
{
	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		sendBuf[i * 2] = new double[sizeEdges[i]];
		getBuf[i * 2] = new double[sizeEdges[i]];
		sendBuf[j] = new double[sizeEdges[i]];
		getBuf[j] = new double[sizeEdges[i]];
		FillBuffer(sendBuf[i * 2], sizeEdges[i], u0);
		FillBuffer(sendBuf[j], sizeEdges[i], u0);

		CSC(cudaMalloc((void**)&deviceSendBuf[i * 2], sizeof(double) * sizeEdges[i]));
		CSC(cudaMalloc((void**)&deviceGetBuf[i * 2], sizeof(double) * sizeEdges[i]));
		CSC(cudaMalloc((void**)&deviceSendBuf[j], sizeof(double) * sizeEdges[i]));
		CSC(cudaMalloc((void**)&deviceGetBuf[j], sizeof(double) * sizeEdges[i]));

		if (!coords[i])
		{
			FillBuffer(getBuf[i * 2], sizeEdges[i], u[i * 2]);
		}
		if (coords[i] == gridProc[i] - 1)
		{
			FillBuffer(getBuf[j], sizeEdges[i], u[j]);
		}
	}
}


void Clear(double* grid, double* newGrid, double** sendBuf, double** getBuf,
		   double* deviceGrid, double* deviceNewGrid, double* deviceMaxValues,
		   double** deviceSendBuf, double** deviceGetBuf)
{
	delete[] grid;
	delete[] newGrid;

	CSC(cudaFree(deviceGrid));
	CSC(cudaFree(deviceNewGrid));
	CSC(cudaFree(deviceMaxValues));

	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		delete[] sendBuf[i * 2];
		delete[] getBuf[i * 2];
		delete[] sendBuf[j];
		delete[] getBuf[j];

		CSC(cudaFree(deviceSendBuf[i * 2]));
		CSC(cudaFree(deviceGetBuf[i * 2]));
		CSC(cudaFree(deviceSendBuf[j]));
		CSC(cudaFree(deviceGetBuf[j]));
	}
}


void GetNeighbours(int* neighb, int* gridProc, int* coords)
{
	int tmp[axes] = {coords[0], coords[1], coords[2]};

	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		--tmp[i];
		neighb[2 * i] = GetRank(tmp, gridProc);
		tmp[i] += 2;
		neighb[j] = GetRank(tmp, gridProc);
		--tmp[i];
	}
}


void Isend_Irecv(MPI_Request* in, MPI_Request* out, double** sendBuf, double** getBuf, int* sizeEdges, int* gridProc, int* coords, int* neighb)
{
	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		if (coords[i] > 0)
		{
			MPI_Isend(sendBuf[i * 2], sizeEdges[i], MPI_DOUBLE,
					  neighb[i * 2], 0, MPI_COMM_WORLD, &out[i * 2]);
      		MPI_Irecv(getBuf[i * 2], sizeEdges[i], MPI_DOUBLE,
      				  neighb[i * 2], 0, MPI_COMM_WORLD, &in[i * 2]);
		}
		if (coords[i] < gridProc[i] - 1) 
		{ 
			MPI_Isend(sendBuf[j], sizeEdges[i], MPI_DOUBLE,
					  neighb[j], 0, MPI_COMM_WORLD, &out[j]);
      		MPI_Irecv(getBuf[j], sizeEdges[i], MPI_DOUBLE,
      				  neighb[j], 0, MPI_COMM_WORLD, &in[j]);
		}
	}
}


__global__
void FillInEdgesKernel(double* grid, double* getBuf, int ax,
						int lim, int nX, int nY, int nZ, int tmpX, int tmpY, int tmpZ)
{
	int offset = gridDim.x * blockDim.x;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	
	// formula to get coords x + (y + z * ny) * nx
	#define idx_in(i, j) ( \
		tmpX * (lim * (nX - 1) + (i + j * nY) * nX) + \
		tmpY * (i + (lim * (nY - 1) + j * nY) * nX) + \
		tmpZ * (i + (j + (lim * (nZ - 1)) * nY) * nX) \
	) \

	int first_n = (ax == 2) ? nY : nZ;
	int second_n = (ax != 0) ? nX : nY;
	int i = 0;
	int j = 0;
	Update_ij(i, j, threadId, second_n);

	while (j < first_n)
	{	
		grid[idx_in(i, j)] = getBuf[i + j * second_n];
		Update_ij(i, j, offset, second_n);
	}
}


__global__
void FillOutEdgesKernel(double* newGrid, double* sendBuf, int ax,
						int lim, int nX, int nY, int nZ, int tmpX, int tmpY, int tmpZ)
{
	int offset = gridDim.x * blockDim.x;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	// formula to get coords x + (y + z * ny) * nx
	#define idx_out(i, j) ( \
		tmpX * (lim * (nX - 3) + 1 + (i + j * nY) * nX) + \
		tmpY * (i + (lim * (nY - 3) + 1 + j * nY) * nX) + \
		tmpZ * (i + (j + (lim * (nZ - 3) + 1) * nY) * nX) \
	) \

	int first_n = (ax == 2) ? nY : nZ;
	int second_n = (ax != 0) ? nX : nY;

	int i = 0;
	int j = 0;
	Update_ij(i, j, threadId, second_n);

	while (j < first_n)
	{
		sendBuf[i + j * second_n] = newGrid[idx_out(i, j)];
		Update_ij(i, j, offset, second_n);
	}
}


__device__
void UpdateCoords(int& i, int& j, int& k, int shift, int nX, int nY)
{
	i += shift;
	while (i > nX)
	{
		i -= nX;
		++j;
	}
	while (j > nY)
	{
		j -= nY;
		++k;
	}
}


__global__
void CalculateNewValuesKernel(double* grid, double* newGrid, double* maxValues,
			   int nX, int nY, int nZ, double hX, double hY, double hZ, double b, int proc)
{
	int offset = gridDim.x * blockDim.x;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int i = 1;
	int j = 1;
	int k = 1;

	UpdateCoords(i, j, k, threadId, nX - 2, nY - 2);
	maxValues[threadId] = 0.0;

	while(k <= nZ - 2)
	{
		double a = (grid[GetPos(i - 1, j, k, nY, nX)] + grid[GetPos(i + 1, j, k, nY, nX)]) / (hX * hX);
		a += (grid[GetPos(i, j - 1, k, nY, nX)] + grid[GetPos(i, j + 1, k, nY, nX)]) / (hY * hY);
		a += (grid[GetPos(i, j, k - 1, nY, nX)] + grid[GetPos(i, j, k + 1, nY, nX)]) / (hZ * hZ);

		newGrid[GetPos(i, j, k, nY, nX)] = a / b;
		maxValues[threadId] = FindMax(grid[GetPos(i, j, k, nY, nX)], newGrid[GetPos(i, j, k, nY, nX)], maxValues[threadId]);
		UpdateCoords(i, j, k, offset, nX - 2, nY - 2);
	}
}



void Start(int* gridProc, int* block, std::string& output,
		   double eps, double* l, double* u,
		   double u0, int numProcs, int rank)
{
	//Широковещательная передача 
	//Broadcasts a message from the process
	// with rank "root" to all other processes of the communicator
	MPI_Bcast(gridProc, axes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(block, axes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(l, axes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(u, directions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	{
		int cntChars = output.size();
		MPI_Bcast(&cntChars, 1, MPI_INT, 0, MPI_COMM_WORLD);

		output.resize(cntChars);

		MPI_Bcast((char*)output.c_str(), cntChars, MPI_CHAR, 0, MPI_COMM_WORLD);
		output.push_back('\0');
	}


	MPI_Request in[directions], out[directions];

	int coords[axes];
	GetCoords(coords, rank, gridProc);

	int neighb[directions];
	GetNeighbours(neighb, gridProc, coords);

	int size = (block[0] + 2) * (block[1] + 2) * (block[2] + 2);
	double* grid = new double[size];
	double* newGrid = new double[size];
	FillBuffer(grid, size, u0);
	FillBuffer(newGrid, size, u0);

	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount);

	// for device
	double* deviceSendBuf[directions];
	double* deviceGetBuf[directions];
	double* deviceMaxValues;
	double* deviceGrid;
	double* deviceNewGrid;

	CSC(cudaMalloc((void**)&deviceMaxValues, sizeof(double) * BlockCount * ThreadsCount));

	CSC(cudaMalloc((void**)&deviceGrid, sizeof(double) * size));

	CSC(cudaMalloc((void**)&deviceNewGrid, sizeof(double) * size));

	CSC(cudaMemcpy(deviceGrid, grid, sizeof(double) * size, cudaMemcpyHostToDevice));

	double* sendBuf[directions];
	double* getBuf[directions];

	int sizeEdges[axes];
	sizeEdges[0] = (block[1] + 2) * (block[2] + 2); // y z
	sizeEdges[1] = (block[0] + 2) * (block[2] + 2); // x z
	sizeEdges[2] = (block[0] + 2) * (block[1] + 2); // x y

	InitBufsEdge(sendBuf, getBuf, deviceSendBuf, deviceGetBuf, sizeEdges, gridProc, coords, u, u0);

	double nX = gridProc[0] * block[0];
	double nY = gridProc[1] * block[1];
	double nZ = gridProc[2] * block[2];

	double hX = (double)(l[0] / nX);
	double hY = (double)(l[1] / nY);
	double hZ = (double)(l[2] / nZ);

	double b = 2 * (1/(hX * hX) + 1/(hY * hY) + 1/(hZ * hZ));
	
	int tmpkernel[axes] = {0};
	int n_X = block[0] + 2, n_Y = block[1] + 2, n_Z = block[2] + 2;

	double maxConvergence;
	double globalMax = 0.0;

	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(deviceMaxValues);
	do {
		maxConvergence = 0.0;
		
		Isend_Irecv(in, out, sendBuf, getBuf, sizeEdges, gridProc, coords, neighb);

		WaitAll(coords, gridProc, in);
		for (int i = 0, j = 1; i < axes; ++i, j += 2)
		{
			CSC(cudaMemcpy(deviceGetBuf[i * 2], getBuf[i * 2], sizeof(double) * sizeEdges[i], cudaMemcpyHostToDevice));
			CSC(cudaMemcpy(deviceGetBuf[j], getBuf[j], sizeof(double) * sizeEdges[i], cudaMemcpyHostToDevice));
		}

		for (int d = 0; d < directions; ++d)
		{
			int ax = d >> 1;
			tmpkernel[ax] = 1;
			int lim = (d & 1);

			FillInEdgesKernel<<<BlockCount, ThreadsCount>>>(deviceGrid, deviceGetBuf[d],
			 														ax, lim, n_X, n_Y, n_Z,
			 														tmpkernel[0], tmpkernel[1], tmpkernel[2]);
			tmpkernel[ax] = 0;
		}
		cudaThreadSynchronize();
		CSC(cudaGetLastError());

		CalculateNewValuesKernel<<<BlockCount, ThreadsCount>>>(deviceGrid, deviceNewGrid, deviceMaxValues,
															   n_X, n_Y, n_Z, hX, hY, hZ, b, rank);
		cudaThreadSynchronize();
		CSC(cudaGetLastError());

		WaitAll(coords, gridProc, out);
		
		for (int d = 0; d < directions; ++d)
		{	
			int ax = d >> 1;
			tmpkernel[ax] = 1;
			int lim = (d & 1);

			FillOutEdgesKernel<<<BlockCount, ThreadsCount>>>(deviceNewGrid, deviceSendBuf[d],
																	 ax, lim, n_X, n_Y, n_Z,
																	 tmpkernel[0], tmpkernel[1], tmpkernel[2]);
			CSC(cudaGetLastError());

			tmpkernel[ax] = 0;
		}
		cudaThreadSynchronize();


		for (int i = 0, j = 1; i < axes; ++i, j += 2)
		{	
			CSC(cudaMemcpy(sendBuf[i * 2], deviceSendBuf[i * 2], sizeof(double) * sizeEdges[i], cudaMemcpyDeviceToHost));
			CSC(cudaMemcpy(sendBuf[j], deviceSendBuf[j], sizeof(double) * sizeEdges[i], cudaMemcpyDeviceToHost));
		}


		maxConvergence = *thrust::max_element(ptr, ptr + BlockCount * ThreadsCount);

		MPI_Allreduce(&maxConvergence, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


		double* tmp = deviceGrid;
	 	deviceGrid = deviceNewGrid;
	 	deviceNewGrid = tmp;
	} while (globalMax >= eps);

	cudaMemcpy(grid, deviceGrid, sizeof(double) * size, cudaMemcpyDeviceToHost);

	WriteOutWithMPI(gridProc, block, output, grid, coords);
	Clear(grid, newGrid, sendBuf, getBuf, deviceGrid, deviceNewGrid, deviceMaxValues, deviceSendBuf, deviceGetBuf);
}



int main(int argc, char *argv[])
{
	std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
	int gridProc[axes], block[axes];
	double l[axes];
	double eps, u0;
	double u[directions];
	std::string output;

	int numProcs, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (!rank) // main process
	{
		//размер сетки процессов
		std::cin >> gridProc[0] >> gridProc[1] >> gridProc[2];
		//размер блока, который будет обрабатываться одним процессом
		std::cin >> block[0] >> block[1] >> block[2];
		std::cin >> output;
		std::cin >> eps;
		std::cin >> l[0] >> l[1] >> l[2];
		//front back down up left right
		std::cin >> u[4] >> u[5] >> u[0] >> u[1] >> u[2] >> u[3];
		std::cin >> u0;
	}
	double time_start;
    if (!rank)
    {
        time_start = MPI_Wtime();
    }

	Start(gridProc, block, output, eps, l, u, u0, numProcs, rank);

	MPI_Finalize();
	double time_end;
    if (!rank)
    {
        time_end = MPI_Wtime();
        std::cout << "TIME: ";
        std::cout << (time_end - time_start) * 1000.0 << "ms" << std::endl;
    }
	return 0;
}
