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

using namespace std;

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
void stencilKernel (???) {

}

// Fill the input parameters and kernel qualifier
void quicksortKernel (???) {

}

// Fill with the code required for the GPU stencil (mem allocation, transfers, kernel launch....)
void stencilGPU (void) {

}

// Fill with the code required for the GPU quicksort (mem allocation, transfers, kernel launch....)
void quicksortGPU (void) {

}

// Fill with the code required for the CPU stencil
void stencilCPU (void) {

}

// Fill with the code required for the CPU quicksort
void quicksortCPU (void) {

}

int main (int argc, char** argv) {
	
	#ifdef D_CPU

	// comment the function that you do not want to execute
	stencilCPU();
	quicksortCPU();

	#elif D_GPU

	// comment the function that you do not want to execute
	stencilGPU();
	quicksortGPU();

	#endif

	return 0;
}
