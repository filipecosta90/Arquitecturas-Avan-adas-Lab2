/**************************************************************
 * 
 * --== Simple CUDA kernel ==--
 * author: ampereira
 * 
 *
 * Fill the rest of the code
 *
 * Insert the functions for time measurement in the correct 
 * sections (i.e. do not account for filling the vectors with random data)
 *
 * Before compile choose the CPU/CUDA version by running the bash command:
 *     export CUDA=yes    or    export CUDA=no
 *
 **************************************************************/
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>

#define TIME_RESOLUTION 1000000	// time measuring resolution (us)
#define NUM_BLOCKS 128
#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK

using namespace std;
timeval t;

long long unsigned cpu_time;
cudaEvent_t start, stop;

void startTime (void) {
	gettimeofday(&t, NULL);
	cpu_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

void stopTime (void) {
	gettimeofday(&t, NULL);
	long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;

	final_time -= cpu_time;

	cout << final_time << " us have elapsed" << endl;
}
#ifdef D_GPU
// These are specific to measure the execution of only the kernel execution - might be useful
void startKernelTime (void) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
}

void stopKernelTime (void) {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << milliseconds << " ms have elapsed for the kernel execution" << endl;
}
#endif

// Fill the input parameters and kernel qualifier
__global__ void stencilKernel (float *in, float *out, int radius) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float value = 0.0f;
	for ( int pos = -radius; pos <= radius; pos++ ){
		value += in[tid+pos];
	}
	out[tid]=value;
}

// Fill the input parameters and kernel qualifier
void quicksortKernel (???) {

}

// Fill with the code required for the GPU stencil (mem allocation, transfers, kernel launch....)
void stencilGPU (void) {
	int bytes = SIZE*sizeof(int);
	float vector[SIZE], output_vector[SIZE];
	float *dev_vector, *dev_output;

	for (unsigned i = 0; i<SIZE; i++){
		vector[i]=(float) rand()/RAND_MAX;
	}

	cudaMalloc((void**)&dev_vector,bytes);
	cudaMalloc((void**)&dev_output,bytes);

	// copy inputs to the device
	cudaMemcpy(dev_vector,&vector,bytes,cudaMemcpyHostToDevice);

	// launch the kernel
	dim3 dimGrid(NUM_BLOCKS);
	dim3 dimBlock(NUM_THREADS_PER_BLOCK);

	startKernelTime();
	stencilKernel<<<dimBlock,dimGrid>>>(dev_vector,dev_output,3);
	stopKernelTime();
	// copy the output to the host
	cudaMemcpy(&output_vector,dev_output,bytes,cudaMemcpyDeviceToHost);

	// free the device memory
	cudaFree(dev_vector);
	cudaFree(dev_output);
}

// Fill with the code required for the GPU quicksort (mem allocation, transfers, kernel launch....)
void quicksortGPU (void) {

}

// Fill with the code required for the CPU stencil
void stencilCPU (int radius) {
	float vector[SIZE], output_vector[SIZE];

	for (unsigned i = 0; i<SIZE; i++){
		vector[i]=(float) rand()/RAND_MAX;
	}

	startTime();
	for ( unsigned i = 0; i < SIZE; i++ ){
		float value = 0.0f;
		for ( int pos = -radius; pos <= radius; pos++ ){
			value += vector[i+pos];
		}
		output_vector[tid]=value;
	}
	stopTime();

	// free the memory
	free(vector);
	free(output_vector);

}

// Fill with the code required for the CPU quicksort
void quickSortCPU(int valor[], int esquerda, int direita){
	int i, j, x, y;
	i = esquerda;
	j = direita;
	x = valor[(esquerda + direita) / 2];

	while(i <= j)
	{
		while(valor[i] < x && i < direita)
		{
			i++;
		}
		while(valor[j] > x && j > esquerda)
		{
			j--;
		}
		if(i <= j)
		{
			y = valor[i];
			valor[i] = valor[j];
			valor[j] = y;
			i++;
			j--;
		}
	}
	if(j > esquerda)
	{
		quickSort(valor, esquerda, j);
	}
	if(i < direita)
	{
		quickSort(valor,  i, direita);
	}
}

int main (int argc, char** argv){
	int radius = 3;
#ifdef D_CPU

	// comment the function that you do not want to execute
	stencilCPU(radius);
	quicksortCPU();

#elif D_GPU
	// comment the function that you do not want to execute
	stencilGPU();

	quicksortGPU();

#endif

	return 0;
}
