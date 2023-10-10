#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <time.h> 
#define BLOCK_SIZE 32        


// Filling the matrices with random numbers.
void FillingMatrices(float* firstMatrix, float* secondMatrix, int sizeMatrix);

// Multiplication of matrices on GPU.
__global__ void MatMultGPU(float* firstMatrix, float* secondMatrix, float* resultMatrix, int sizeMatrix);

// Multiplication of matrices on CPU.
float MatMulCPU(float* firstMatrix, float* secondMatrix, float* resultMatrix, int sizeMatrix);

// Checking the equality of matrices.
bool CheckEquality(float* firstMatrix, float* secondMatrix, int sizeMatrix);

int main(int argc, char* argv[])
{
	// Matrix size is sizeMatrix * sizeMatrix.
	int sizeMatrix = 512;
	std::cout << "Matrix size: " << sizeMatrix << std::endl;

	// Memory allocation for matrices.
	float* firstMatrix = new float[sizeMatrix * sizeMatrix];
	float* secondMatrix = new float[sizeMatrix * sizeMatrix];
	float* cpuResultMatrix = new float[sizeMatrix * sizeMatrix];
	float* gpuResultMatrix = new float[sizeMatrix * sizeMatrix];

	// Filling matrices with random numbers.
	FillingMatrices(firstMatrix, secondMatrix, sizeMatrix);

	// Memory allocation for matrices for CUDA.
	float* cudaFirstMatrix, * cudaSecondMatrix, * cudaResultMatrix;
	cudaMalloc((void**)&cudaFirstMatrix, sizeMatrix * sizeMatrix * sizeof(float));
	cudaMalloc((void**)&cudaSecondMatrix, sizeMatrix * sizeMatrix * sizeof(float));
	cudaMalloc((void**)&cudaResultMatrix, sizeMatrix * sizeMatrix * sizeof(float));

	// Event handler.
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Number of threads and blocks.
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(sizeMatrix / threads.x, sizeMatrix / threads.y);

	// Records the start event.
	cudaEventRecord(start, 0);

	// Сopy matrices to the device.
	cudaMemcpy(cudaFirstMatrix, firstMatrix, sizeMatrix * sizeMatrix * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaSecondMatrix, secondMatrix, sizeMatrix * sizeMatrix * sizeof(float), cudaMemcpyHostToDevice);

	// Multiplication of matrices on GPU.
	MatMultGPU <<< blocks, threads >>> (cudaFirstMatrix, cudaSecondMatrix, cudaResultMatrix, sizeMatrix);

	// Get the result of multiplication.
	cudaMemcpy(gpuResultMatrix, cudaResultMatrix, sizeMatrix * sizeMatrix * sizeof(float), cudaMemcpyDeviceToHost);

	// Records the stop event.
	cudaEventRecord(stop, 0);
	// Waits for the stop event to complete.
	cudaEventSynchronize(stop);

	float gpuTime = 0.0f;
	cudaEventElapsedTime(&gpuTime, start, stop);
	
	std::cout << "\nGPU time is " << std::fixed << std::setprecision(2) << gpuTime << " ms\n";
	
	// Multiplication of matrices on CPU.
	float cpuTime = MatMulCPU(firstMatrix, secondMatrix, cpuResultMatrix, sizeMatrix);
	std::cout << "\nCPU time is " << std::fixed << std::setprecision(2) << cpuTime << " ms\n";

	std::cout << "\nAre the matrices equal: ";
	CheckEquality(cpuResultMatrix, gpuResultMatrix, sizeMatrix) ? std::cout << "yes" : std::cout << "no";

	cudaFreeHost(firstMatrix);
	cudaFreeHost(secondMatrix);
	cudaFreeHost(cpuResultMatrix);
	cudaFreeHost(gpuResultMatrix);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(cudaFirstMatrix);
	cudaFree(cudaSecondMatrix);
	cudaFree(cudaResultMatrix);

	return 0;
}

// Filling the matrices with random numbers.
void FillingMatrices(float* firstMatrix, float* secondMatrix, int sizeMatrix)
{
	srand(time(0));
	for (int i = 0; i < sizeMatrix; i++)
	{
		for (int j = 0; j < sizeMatrix; j++)
		{
			firstMatrix[i * sizeMatrix + j] = rand() % 1000;
			secondMatrix[i * sizeMatrix + j] = rand() % 1000;
		}
	}
}

// Multiplication of matrices on GPU.
__global__ void MatMultGPU(float* firstMatrix, float* secondMatrix, float* resultMatrix, int sizeMatrix)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0f;
	int indexFirstMatrix = sizeMatrix * blockDim.y * blockIdx.y + sizeMatrix * threadIdx.y;
	int indexSecondMatrix = blockDim.x * blockIdx.x + threadIdx.x;
	int indexResultMatrix = sizeMatrix * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;

	// calculating the element
	for (int k = 0; k < sizeMatrix; k++)
	{
		sum += firstMatrix[indexFirstMatrix + k] * secondMatrix[indexSecondMatrix + k * sizeMatrix];
	}
	resultMatrix[indexResultMatrix + sizeMatrix * ty + tx] = sum;
}

// Multiplication of matrices on CPU.
float MatMulCPU(float* firstMatrix, float* secondMatrix, float* resultMatrix, int sizeMatrix)
{
	clock_t start = clock();
	for (int i = 0; i < sizeMatrix; i++)
	{
		for (int j = 0; j < sizeMatrix; j++)
		{
			resultMatrix[i * sizeMatrix + j] = 0;
			for (int k = 0; k < sizeMatrix; k++)
			{
				resultMatrix[i * sizeMatrix + j] += firstMatrix[i * sizeMatrix + k] * secondMatrix[k * sizeMatrix + j];
			}
		}
	}

	clock_t end = clock();
	// Time is in ms.
	float time = ((float)(end - start) / CLOCKS_PER_SEC) * 1000;

	return time;
}

// Checking the equality of matrices.
bool CheckEquality(float* firstMatrix, float* secondMatrix, int sizeMatrix)
{
	/*
		The memcmp function from the string library compares the memory blocks of two matrices and
		returns 0 if they are equal
	*/
	int numBytes = sizeMatrix * sizeMatrix * sizeof(float);
	return memcmp(firstMatrix, secondMatrix, numBytes) == 0;
}
