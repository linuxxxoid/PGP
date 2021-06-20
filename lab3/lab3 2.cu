
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>


#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


__constant__ float constAVG[32][3];
__constant__ float constNormaAVG[32];


__device__ void FormulaComputation(float* res, int numClasses, uchar4 curPixel)
{
	float rgb[3];
	float tmp[3];
	float sum, denominator;

	float normaPix = sqrt((float)(curPixel.x * curPixel.x + curPixel.y * curPixel.y + curPixel.z * curPixel.z));

	for (int curClass = 0; curClass < numClasses; ++curClass)
	{
		rgb[0] = curPixel.x * constAVG[curClass][0];
		rgb[1] = curPixel.y * constAVG[curClass][1];
		rgb[2] = curPixel.z * constAVG[curClass][2];

		denominator = normaPix * constNormaAVG[curClass];
		rgb[0] /= denominator;
		rgb[1] /= denominator;
		rgb[2] /= denominator;
		
		sum = 0.0;
		for (int i = 0; i < 3; ++i) tmp[i] = 0.0;

		for (int i = 0; i < 3; ++i)
		{
			tmp[i] += rgb[0];
			tmp[i] += rgb[1];
			tmp[i] += rgb[2];
			sum += tmp[i];
		}
		res[curClass] = sum;
	}
}
			

__device__ int ArgMax(float* arr, int numClasses)
{
	float maxValue = arr[0];
	int maxPoint = 0;

	for (int i = 0; i < numClasses; ++i)
	{
		if (arr[i] > maxValue)
		{
			maxValue = arr[i];
			maxPoint = i;
		}
	}
	return maxPoint;
}


__global__ void SpectralAngleMethod(uchar4* pixels, int width, int height, int numClasses) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int xOffset = blockDim.x * gridDim.x;
	int yOffset = blockDim.y * gridDim.y;

	uchar4 curPixel;
	float res[32];

	for (int x = idx; x < width; x += xOffset)
	{
		for (int y = idy; y < height; y += yOffset)
		{
			curPixel = pixels[y * width + x];
			FormulaComputation(res, numClasses, curPixel);
			pixels[y * width + x].w = ArgMax(res, numClasses);
		}
	}
}


int main(int argc, const char* argv[])
{
	std::string input, output;
	int width, height, numClasses, numPixels;
	uchar4* pixels;

	std::cin >> input >> output >> numClasses;

	int2 coordinate;
	std::vector<std::vector<int2>> samples(numClasses);

	for (int i = 0; i < numClasses; ++i)
	{
		std::cin >> numPixels;
		for (int j = 0; j < numPixels; ++j)
		{
			std::cin >> coordinate.x >> coordinate.y;
			samples[i].emplace_back(coordinate);
		}
	}

	FILE* file;
	if ((file = fopen(input.c_str(), "rb")) == NULL)
	{
	    std::cerr << "ERROR: something wrong with opening the file!\n";
        exit(0);
	}
	else
	{
		fread(&width, sizeof(int), 1, file);
		fread(&height, sizeof(int), 1, file);
		if (width * height > 400000000)
		{
			std::cerr << "ERROR: incorrect input.\n";
			exit(0);
		}
		pixels = new uchar4[width * height];
		fread(pixels, sizeof(uchar4), width * height, file);

		fclose(file);
	}

	int numChannels = 3; // rgb
	int maxElems = 32;

	float avg[maxElems][numChannels];

	for (int i = 0; i < numClasses; ++i)
	{
		avg[i][0] = 0.0;
		avg[i][1] = 0.0;
		avg[i][2] = 0.0;

		numPixels = samples[i].size();
		for (int j = 0; j < numPixels; ++j)
		{
			coordinate.x = samples[i][j].x;
			coordinate.y = samples[i][j].y;
			avg[i][0] += pixels[coordinate.y * width + coordinate.x].x;
			avg[i][1] += pixels[coordinate.y * width + coordinate.x].y;
			avg[i][2] += pixels[coordinate.y * width + coordinate.x].z;
		}
		avg[i][0] /= numPixels;
		avg[i][1] /= numPixels;
		avg[i][2] /= numPixels;
	}


    float normaAvg[32];
	for (int i = 0; i < numClasses; ++i)
	{
		normaAvg[i] = std::sqrt(avg[i][0] * avg[i][0] + avg[i][1] * avg[i][1] + avg[i][2] * avg[i][2]);
	}

	CSC(cudaMemcpyToSymbol(constAVG, avg, sizeof(float) * maxElems * numChannels));
	CSC(cudaMemcpyToSymbol(constNormaAVG, normaAvg, sizeof(float) * maxElems));

	uchar4* deviceRes;
	CSC(cudaMalloc(&deviceRes, sizeof(uchar4) * width * height));
	CSC(cudaMemcpy(deviceRes, pixels, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));

	int xThreadCount = 16;
	int yThreadCount = 16;

	int xBlockCount = 16;
	int yBlockCount = 16;

    dim3 blockCount = dim3(xBlockCount, yBlockCount);
    dim3 threadsCount = dim3(xThreadCount, yThreadCount);

    SpectralAngleMethod<<<blockCount, threadsCount>>>(deviceRes, width, height, numClasses);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(pixels, deviceRes, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
	

	if ((file = fopen(output.c_str(), "wb")) == NULL)
	{
		std::cerr << "ERROR: something wrong with opening the file.";
        exit(0);
	}
	else
	{
		fwrite(&width, sizeof(int), 1, file);
		fwrite(&height, sizeof(int), 1, file);
		fwrite(pixels, sizeof(uchar4), width * height, file);

		fclose(file);
	}

	CSC(cudaFree(deviceRes));

	delete[] pixels;
	return 0;
}