
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>


using namespace std;


#define N 512
#define M 512 //threads per block

__global__ void add(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
		c[index] = a[index] + b[index];
}void sequential_ints(int* a, int size)
{
	for (int i = 0; i < size; i++)
		a[i] = i;
}

int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N * sizeof(int);
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); 
	sequential_ints(a, N);
	b = (int *)malloc(size); 
	sequential_ints(b, N);
	c = (int *)malloc(size);
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU
	add << <(N + M - 1) / M, M >> >(d_a, d_b, d_c, N);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	// Cleanup
	for(int i = 0; i < N; i++)
		cout << a[i] << ", " << b[i] << ", " << c[i] << endl;
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
