#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "mpi.h"



const int axes = 3;
const int directions = 6;


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


double FindMax(double a, double b, double max)
{
	if (fabs(a - b) > max)
	{
		max = fabs(a - b);
	}
	return max;
}


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
	

void WriteOut(std::string& output, int* gridProc, int* block, int* coords, double* grid, double* exchangeBuf, int rank)
{
	MPI_Status status;
	FILE* out;
	int tmp[axes];

	if (!rank)
	{
		out = fopen(output.c_str(), "w");
	}

	for(int procZ = 0; procZ < gridProc[2]; ++procZ)
	{
        tmp[2] = procZ;
        for (int k = 1; k <= block[2]; ++k)
        {
            for (int procY = 0; procY < gridProc[1]; ++procY)
            {
                tmp[1] = procY;
                for (int j = 1; j <= block[1]; ++j)
                {
                    for (int procX = 0; procX < gridProc[0]; ++procX)
                    {
                        tmp[0] = procX;
                        if (!rank)
                        {
                            if (coords[2] == procZ && coords[1] == procY && coords[0] == procX)
                            {
                                Printer(out, &grid[GetPos(1, j, k, block[1] + 2, block[0] + 2)], block[0]);
                            }
                            else
                            {
                                int rank = GetRank(tmp, gridProc);
                                MPI_Recv(exchangeBuf, block[0], MPI_DOUBLE, rank,
                                    	 0, MPI_COMM_WORLD, &status);
                                Printer(out, exchangeBuf, block[0]);
                            }
                        }
                        else
                        {
                            if (coords[0] == procX && coords[1] == procY && coords[2] == procZ)
                            {
                                MPI_Send(&grid[GetPos(1, j, k, block[1] + 2, block[0] + 2)],
                                		 block[0], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // Bsend may be better
                            }
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD); // syncronize
                }
            }
        }
    }
    if (!rank)
	{
		fclose(out);
	}
}


void FillBuffer(double* buf, int size, double val)
{
	for (int i = 0; i < size; ++i)
	{
		buf[i] = val;
	}
}


void InitBufsEdge(double** sendBuf, double** getBuf, int* sizeEdges, int* gridProc, int* coords, double* u, double u0)
{
	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		sendBuf[i * 2] = new double[sizeEdges[i]];
		getBuf[i * 2] = new double[sizeEdges[i]];
		sendBuf[j] = new double[sizeEdges[i]];
		getBuf[j] = new double[sizeEdges[i]];
		FillBuffer(sendBuf[i * 2], sizeEdges[i], u0);
		FillBuffer(sendBuf[j], sizeEdges[i], u0);

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


void Clear(double* grid, double* newGrid, double** sendBuf, double** getBuf)
{
	delete[] grid;
	delete[] newGrid;

	for (int i = 0, j = 1; i < axes; ++i, j += 2)
	{
		delete[] sendBuf[i * 2];
		delete[] getBuf[i * 2];
		delete[] sendBuf[j];
		delete[] getBuf[j];
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


void FillInEdges(double* grid, double** getBuf, int* block)
{
	int tmp[axes] = {0};
	++tmp[0];
	--tmp[0];
	int nX = block[0] + 2, nY = block[1] + 2, nZ = block[2] + 2;

	for (int d = 0; d < directions; ++d)
	{
		int ax = d >> 1;
		tmp[ax] = 1;
		int lim = (d & 1);
		// formula to get coords x + (y + z * ny) * nx
		auto idx = [nX, nY, nZ, tmp, lim](int i, int j)
		{
			return tmp[0] * (lim * (nX - 1) + (i + j * nY) * nX) +
			tmp[1] * (i + (lim * (nY - 1) + j * nY) * nX) +
			tmp[2] * (i + (j + (lim * (nZ - 1)) * nY) * nX);
		};

		int first_n = (ax == 2) ? nY : nZ;
		int second_n = (ax != 0) ? nX : nY;

		for (int j = 0; j < first_n; ++j)
		{
			for (int i = 0; i < second_n; ++i)
			{
				grid[idx(i, j)] = getBuf[d][i + j * second_n];
			}
		}
		tmp[ax] = 0;
	}
}



void FillOutEdges(double* newGrid, double** sendBuf, int* block)
{
	int tmp[axes] = {0};
	++tmp[0];
	--tmp[0];
	int nX = block[0] + 2, nY = block[1] + 2, nZ = block[2] + 2;

	for (int d = 0; d < directions; ++d)
	{
		int ax = d >> 1;
		tmp[ax] = 1;
		int lim = (d & 1);
		// formula to get coords x + (y + z * ny) * nx
		auto idx = [nX, nY, nZ, tmp, lim](int i, int j)
		{
			return tmp[0] * (lim * (nX - 3) + 1 + (i + j * nY) * nX) +
			tmp[1] * (i + (lim * (nY - 3) + 1 + j * nY) * nX) +
			tmp[2] * (i + (j + (lim * (nZ - 3) + 1) * nY) * nX);
		};

		int first_n = (ax == 2) ? nY : nZ;
		int second_n = (ax != 0) ? nX : nY;

		for (int j = 0; j < first_n; ++j)
		{
			for (int i = 0; i < second_n; ++i)
			{
				sendBuf[d][i + j * second_n] = newGrid[idx(i, j)];
			}
		}
		tmp[ax] = 0;
	}
}


void Start(int* gridProc, int* block, std::string& output,
		   double eps, double* l, double* u,
		   double u0, int numProcs, int rank)
{
	MPI_Bcast(gridProc, axes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(block, axes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(l, axes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(u, directions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	MPI_Request in[directions], out[directions];

	int coords[axes];
	GetCoords(coords, rank, gridProc);

	int neighb[directions];
	GetNeighbours(neighb, gridProc, coords);

	double* grid = new double[(block[0] + 2) * (block[1] + 2) * (block[2] + 2)];
	double* newGrid = new double[(block[0] + 2) * (block[1] + 2) * (block[2] + 2)];
	FillBuffer(grid, (block[0] + 2) * (block[1] + 2) * (block[2] + 2), u0);
	FillBuffer(newGrid, (block[0] + 2) * (block[1] + 2) * (block[2] + 2), u0);

	double* sendBuf[directions];
	double* getBuf[directions];

	int sizeEdges[axes];
	sizeEdges[0] = (block[1] + 2) * (block[2] + 2); // y z
	sizeEdges[1] = (block[0] + 2) * (block[2] + 2); // x z
	sizeEdges[2] = (block[0] + 2) * (block[1] + 2); // x y

	InitBufsEdge(sendBuf, getBuf, sizeEdges, gridProc, coords, u, u0);


	double nX = gridProc[0] * block[0];
	double nY = gridProc[1] * block[1];
	double nZ = gridProc[2] * block[2];


	double hX = (double)(l[0] / nX);
	double hY = (double)(l[1] / nY);
	double hZ = (double)(l[2] / nZ);

	double maxConvergence;
	double a, b, globalMax = 0.0;

	do {
		maxConvergence = 0.0;
		Isend_Irecv(in, out, sendBuf, getBuf, sizeEdges, gridProc, coords, neighb);

		WaitAll(coords, gridProc, in);

		FillInEdges(grid, getBuf, block);

		for (int k = 1; k <= block[2]; ++k)
		{
			for (int j = 1; j <= block[1]; ++j)
			{
				for (int i = 1; i <= block[0]; ++i)
				{
					a = (grid[GetPos(i - 1, j, k, block[1] + 2, block[0] + 2)] + grid[GetPos(i + 1, j, k, block[1] + 2, block[0] + 2)]) / (hX * hX);
					a += (grid[GetPos(i, j - 1, k, block[1] + 2, block[0] + 2)] + grid[GetPos(i, j + 1, k, block[1] + 2, block[0] + 2)]) / (hY * hY);
					a += (grid[GetPos(i, j, k - 1, block[1] + 2, block[0] + 2)] + grid[GetPos(i, j, k + 1, block[1] + 2, block[0] + 2)]) / (hZ * hZ);

					b = 2 * (1/(hX * hX) + 1/(hY * hY) + 1/(hZ * hZ));
					newGrid[GetPos(i, j, k, block[1] + 2, block[0] + 2)] = a / b;
					maxConvergence = FindMax(grid[GetPos(i, j, k, block[1] + 2, block[0] + 2)], newGrid[GetPos(i, j, k, block[1] + 2, block[0] + 2)], maxConvergence);
				}
			}
		}
		 
		WaitAll(coords, gridProc, out);

		FillOutEdges(newGrid, sendBuf, block);
		
		MPI_Allreduce(&maxConvergence, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		double* tmp = grid;
	 	grid = newGrid;
	 	newGrid = tmp;

	} while (globalMax >= eps);

	WriteOut(output, gridProc, block, coords, grid, newGrid, rank);
	Clear(grid, newGrid, sendBuf, getBuf);
}



int main(int argc, char *argv[])
{
	//axes=3
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
		std::cin >> gridProc[0] >> gridProc[1] >> gridProc[2];
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
