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

			stencilCPU(3);
			quicksortCPU();
		return 0;
}

