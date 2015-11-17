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

// Fill the input parameters and kernel qualifier
__global__ void stencilKernel (float *in, float *out, int radius) {

	for ( int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < SIZE; tid += blockIdx.x + blockDim.x ){
		float value = 0.0f;
		for ( int pos = -radius; pos <= radius; pos++ ){
			value += in[tid+pos];
		}
		out[tid]=value;
	}
}

/*
// Fill the input parameters and kernel qualifier
void quicksortKernel (???) {

}
 */

// Fill with the code required for the GPU stencil (mem allocation, transfers, kernel launch....)
void stencilGPU (void) {

	int bytes = SIZE*sizeof(int);
	float vector[SIZE], output_vector[SIZE];
	float *dev_vector, *dev_output;

	// create random vector
	for (unsigned i = 0; i<SIZE; i++){
		vector[i]=(float) rand()/RAND_MAX;
	}

	// malloc memmory device
	cudaMalloc((void**)&dev_vector,bytes);
	cudaMalloc((void**)&dev_output,bytes);
	startKernelTime();
	// copy inputs to the device
	cudaMemcpy(dev_vector,&vector,bytes,cudaMemcpyHostToDevice);

	// launch the kernel
	dim3 dimGrid(NUM_BLOCKS);
	dim3 dimBlock(NUM_THREADS_PER_BLOCK);


	stencilKernel<<<dimBlock,dimGrid>>>(dev_vector,dev_output,3);
	// copy the output to the host
	cudaMemcpy(&output_vector,dev_output,bytes,cudaMemcpyDeviceToHost);
	stopKernelTime();

	// free the device memory
	cudaFree(dev_vector);
	cudaFree(dev_output);
}
/*
// Fill with the code required for the GPU quicksort (mem allocation, transfers, kernel launch....)
void quicksortGPU (void) {

}*/

int main (int argc, char** argv){

	stencilGPU();
	return 0;
}

