#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <iterator>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <cstring>
using namespace std::chrono;
//countingSort sort function

static void countingSort(unsigned char* items, unsigned char* output, int len) {

	//Range for counting array
	const unsigned int RANGE = 256;
	std::vector<int> count(RANGE,0);

	//Counting each elements numbers of time they appered
	for(unsigned int i = 0; i < len ; ++i) {
		++count[items[i]];
	}

	//Modifiying each count such that each element at each index stors the sum of previous counts
	for(auto it = std::next(count.begin()) ; it != count.end() ; it = std::next(it)) {
		*it += *(std::prev(it));
	}

	//Ouput variable where the sorted element will be stored
	for(unsigned int i = 0; i < len ; ++i) {
		output[count[items[i]] - 1] = items[i];
		--count[items[i]];
	}

	//copying variable to the original array
	//std::copy(output.begin(),output.end(),items.begin());

}


template<class T>
inline void printElement(const T& items,const std::string& heading) {
	std::cout << heading <<std::endl;
	//Printing Element to standart output
	std::copy(items.begin(),items.end(),
		std::ostream_iterator<typename T::value_type>(std::cout," "));
	std::cout << std::endl;
}
int main() {
	//std::array<float, 10> elem = {9.2,6.4,2.33,1.0,23.23,4.99,6.53,7.01,4.0,3.2};
	int size;
	freopen(NULL, "rb", stdin);
	fread(&size, sizeof(int), 1, stdin);

    unsigned char* data = new unsigned char[size];
    unsigned char* res = new unsigned char[size];

	fread(data, sizeof(unsigned char), size, stdin);
	fclose(stdin);

	//printElement(data, "Unsorted array:");
	auto start = steady_clock::now();
	countingSort(data, res, size);
	auto end = steady_clock::now();
    std::cout << "Time: " << ((double)duration_cast<microseconds>(end - start).count()) / 1000.0 << "ms" << std::endl;

	//printElement(data,"Sorted Array:");
	
}