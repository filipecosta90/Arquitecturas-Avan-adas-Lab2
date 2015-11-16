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
/*
// Fill with the code required for the GPU quicksort (mem allocation, transfers, kernel launch....)
void quicksortGPU (void) {

}*/

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

    output_vector[i]=value;
  }
  stopTime();

}

void quick(float vet[], int esq, int dir){
  int pivo = esq,i,ch,j;

  for(i=esq+1;i<=dir;i++){
    j = i;
    if(vet[j] < vet[pivo]){
      ch = vet[j];
      while(j > pivo){
        vet[j] = vet[j-1];
        j--;
      }
      vet[j] = ch;
      pivo++;
    }
  }

  if(pivo-1 >= esq){
    quick(vet,esq,pivo-1);
  }

  if(pivo+1 <= dir){
    quick(vet,pivo+1,dir);
  }
}

void quicksortCPU() {

  float vector[SIZE];

  for (unsigned i = 0; i<SIZE; i++){
    vector[i]=(float) rand()/RAND_MAX;
  }

  //start timer
  startTime();

  //do the work
  quick( vector, 0 , SIZE-1);

  //stop timer
  stopTime(); 

}
int main (int argc, char** argv){
#ifdef D_CPU

  // comment the function that you do not want to execute
  stencilCPU(3);
  quicksortCPU();

#elif D_GPU
  // comment the function that you do not want to execute
  stencilGPU();
  //quicksortGPU();

#endif

  return 0;
}
