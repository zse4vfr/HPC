#ifndef __CUDACC__  
    #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <time.h> 
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 512


// The sum of the elements of the vector on CPU.
float SumOnCPU(int* vec, int number, int &sum);

// The sum of the elements of the vector on GPU.
__global__ void SumOnGPU(int* vec, int* result, int number);


int main(int argc, char* argv[])
{
	int resultCPU = 0, resultGPU = 0;
	srand(time(NULL));
	int number = 10000000;
	int n2b = number * sizeof(int);

	// Memory allocation on the host.
	int* vec = (int*)calloc(number, sizeof(int));
	int* resultVec = (int*)calloc(number, sizeof(int));

	// Filling random numbers.
	for (int i = 0; i < number; ++i) 
	{
		vec[i] = (int)rand() % 100;
	}

	// Memory allocation on the device.
	int* vecDev = NULL;
	cudaMalloc((void**)&vecDev, n2b);
	int* resultVecDev= NULL;
	cudaMalloc((void**)&resultVecDev, n2b);

	// Creating event handlers.
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Copying data from the host to the device.
	cudaMemcpy(vecDev, vec, n2b, cudaMemcpyHostToDevice);

	// Records the start event.
	cudaEventRecord(start, 0);

	// The sum of the elements of the vector on GPU.
	SumOnGPU << < 1, BLOCK_SIZE >> > (vecDev, resultVecDev, number);

	// Get tre result.
	cudaMemcpy(resultVec, resultVecDev, n2b, cudaMemcpyDeviceToHost);

	// Records the stop event.
	cudaEventRecord(stop, 0);
	// Waits for the stop event to complete.
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);

	float cpuTime = SumOnCPU(vec, number, resultCPU);
	resultGPU = resultVec[0];

	std::cout << "\nGPU time is " << std::fixed << std::setprecision(2) << gpuTime << " ms\n";
	std::cout << "\nCPU time is " << std::fixed << std::setprecision(2) << cpuTime << " ms\n";
	std::cout << "\nThe sum of the elements of the vector on CPU is " << resultCPU << "\n";
	std::cout << "\nThe sum of the elements of the vector on GPU is " << resultGPU << "\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(vecDev);
	cudaFree(resultVecDev);
	free(vec);
	free(resultVec);

	return 0;
}

// The sum of the elements of the vector on GPU.
__global__ void SumOnGPU(int* vec, int* result, int number)
{
	// Shared memory allocation.
	__shared__ int shArray[BLOCK_SIZE];
	// Number of thread.
	unsigned int threadId = threadIdx.x;

	// Partial amounts.
	int partial = 0;
	for (int i = threadId; i < number; i += BLOCK_SIZE)
	{
		partial += vec[i];
	}

	shArray[threadId] = partial;
	__syncthreads();

	for (unsigned int j = 1; j < blockDim.x; j *= 2) 
	{
		if (threadId % (2 * j) == 0) 
		{
			shArray[threadId] += shArray[threadId + j];
		}
		__syncthreads();
	}

	if (!threadId) 
	{
		result[blockIdx.x] += shArray[0];
	}
}

//The sum of the elements of the vector on CPU.
float SumOnCPU(int* vec, int number, int &sum)
{
	clock_t start = clock();

	for (int i = 0; i < number; i++)
	{
		sum += vec[i];
	}

	clock_t end = clock();
	float time = ((float)(end - start) / CLOCKS_PER_SEC) * 1000;
	return time;
}