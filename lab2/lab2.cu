#include <iostream>
#include <stdio.h>
#include <stdlib.h>


texture<uchar4, 2, cudaReadModeElementType> Texture2D;


void checkCudaError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "ERROR: %s: %s!\n", msg, cudaGetErrorString(err));
        exit(0);
    }
}



__global__ void SSAA(uchar4 *colorPixels, int width, int height, int proportionWidth, int proportionHeight)
{
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;
	int xOffset = blockDim.x * gridDim.x;
	int yOffset = blockDim.y * gridDim.y;
	int numSample = proportionWidth * proportionHeight;

	/*
	colorPixels = colorSample_0 + colorSample_1 + ... + colorSample_n-1   SUM(colorSample_i)
		          ----------------------------------------------------- = ------------------
	                            numSample                                   numSample
	 
	 colorPixels - the final color of the pixel,
	 numSample - the number of samples per pixel,
	 colorSample_i - color of the i-th sample.
	*/

	for (int col = xId; col < width; col += xOffset)
	{
		for (int row = yId; row < height; row += yOffset)
		{
			int3 colorSample;
			colorSample.x = 0;
			colorSample.y = 0;
			colorSample.z = 0;
			for (int i = 0; i < proportionWidth; ++i)
			{
				for (int j = 0; j < proportionHeight; ++j)
				{
					uchar4 pix = tex2D(Texture2D, col * proportionWidth + i, row * proportionHeight + j);
					colorSample.x += pix.x;
					colorSample.y += pix.y;
					colorSample.z += pix.z;
				}
			}
			colorSample.x /= numSample; 
			colorSample.y /= numSample;
			colorSample.z /= numSample;
            // Write to global memory
			colorPixels[col + row * width] = make_uchar4(colorSample.x, colorSample.y, colorSample.z, 0);
		}
	}
}



int main(int argc, const char* argv[])
{	
	std::string input, output;
	int widthNew, heightNew, width, height;
	uchar4 *pixels;
	std::cin >> input >> output >> widthNew >> heightNew;

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
		if (width >= 65536 || width < 0 || height < 0 || height >= 65536)
		{
			std::cerr << "ERROR: incorrect input.\n";
			exit(0);
		}
		pixels = new uchar4[width * height];
		fread(pixels, sizeof(uchar4), width * height, file);

		fclose(file);
	}

	int proportionWidth = width / widthNew;
	int proportionHeight = height / heightNew;

    // Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaArray *array;

	cudaMallocArray(&array, &channelDesc, width, height);
	checkCudaError("Malloc array");

	cudaMemcpyToArray(array, 0, 0, pixels, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice);
	checkCudaError("Memcpy array");

	// set texture parameters
	Texture2D.addressMode[0] = cudaAddressModeClamp;
	Texture2D.addressMode[1] = cudaAddressModeClamp;
	Texture2D.channelDesc = channelDesc;
	Texture2D.filterMode = cudaFilterModePoint;;
	Texture2D.normalized = false; // access with normalized texture coordinates

	// Bind the array to the texture
	cudaBindTextureToArray(Texture2D, array, channelDesc);
	checkCudaError("Bind");

	uchar4 *deviceRes;
	cudaMalloc(&deviceRes, sizeof(uchar4) * widthNew * heightNew);
	checkCudaError("Malloc");


    int xThreadCount = 16;
    int yThreadCount = 16;

	int xBlockCount = 16;
	int yBlockCount = 16;

 
    dim3 blockCount = dim3(xBlockCount, yBlockCount);
    dim3 threadsCount = dim3(xThreadCount, yThreadCount);

	SSAA<<<blockCount, threadsCount>>>(deviceRes, widthNew, heightNew, proportionWidth, proportionHeight);
	checkCudaError("Kernel invocation");

	cudaMemcpy(pixels, deviceRes, sizeof(uchar4) * widthNew * heightNew, cudaMemcpyDeviceToHost);
	checkCudaError("Memcpy");
	
	if ((file = fopen(output.c_str(), "wb")) == NULL)
	{
		std::cerr << "ERROR: something wrong with opening the file.";
        exit(0);
	}
	else
	{
		fwrite(&widthNew, sizeof(int), 1, file);
		fwrite(&heightNew, sizeof(int), 1, file);
		fwrite(pixels, sizeof(uchar4), widthNew * heightNew, file);

		fclose(file);
	}


	cudaUnbindTexture(Texture2D);
	checkCudaError("Unbind");

	cudaFreeArray(array);
	checkCudaError("Free");
	
	cudaFree(deviceRes);
	checkCudaError("Free");

	delete[] pixels;
	return 0;
}