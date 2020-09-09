#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, const char* argv[])
{	
	std::string input = "./test";
	std::string output;
	int widthNew = 2;
	int heightNew = 2;
	

	FILE* file;
	if ((file = fopen(input.c_str(), "rb")) == NULL)
	{
		std::cerr << "ERROR: something wrong with opening the file!\n";
		exit(0)
	}
	else
	{
		fread(&width, sizeof(int), 1, file);
		fread(&height, sizeof(int), 1, file);
		if (width >= 65536 || width < 0 || height < 0 || height >= 65536)
		{
			std::cerr << "ERROR: incorrect input!\n";
		}
		//fread(&height, 1, sizeof(int), file);
		printf("%i %i\n", width, height);
		pixels = new uchar4[width * height];
		fread(pixels, sizeof(uchar4), width * height, file);

		fclose(file);
	}

		return 0;
}
