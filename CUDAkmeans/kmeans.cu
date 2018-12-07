
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <vector>
//#include <float.h>


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


typedef vector<double*> Data;
typedef vector<double*> Centroids;
typedef vector<int> Output;

class kmeans {
public:
	kmeans(Data* data, int n, int dim) {
		this->data = data;
		this->n = n;
		this->dim = dim;
		this->k = 3;
		this->maxIter = 1000;
	}


	int* cluster(int k, int maxIter, Output out) {
		this->k = k;
		this->maxIter = maxIter;
		randCentroids();	//make this parallel
		int iter = 0;
		bool converged = false;
		while (!converged && iter < maxIter) {
			nearestCentroids();	//make this parallel
			converged = calcCentroids();	//make this parallel
			iter++;
		}

	}

	// fields
	Data* data;
	Centroids centroids;
	int n;
	int dim;
	int k;
	int maxIter;

private:
	double fRand(double min, double max) {
		double f = (double)rand() / RAND_MAX;
		return min + f * (max - min);
	}
	/* finds nearest centroids, must make parallel*/
	Output* nearestCentroids() {




	}

	/*recaclulates new centroids, must make parallel*/
	bool calcCentroids() {

	}
	/** I think we can paralellize this*/
	void randCentroids() {
		//get range of dataset
		vector<double> min = { DBL_MAX, DBL_MAX, DBL_MAX };
		vector<double> max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim; j++) {
				if (data->at(i)[j] > max.at(j))
					max.at(j) = data->at(i)[j];
				if (data->at(i)[j] < min.at(j))
					min.at(j) = data->at(i)[j];
			}
		}

		double* tmp;
		for (int i = 0; i < k; i++) {
			tmp = new double[dim];
			for (int j = 0; j < dim; j++) {
				tmp[j] = fRand(min.at(k), max.at(k));
			}
		}
		centroids.push_back(tmp);
	}

};

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
	add << <(N + M - 1) / M, M >> > (d_a, d_b, d_c, N);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	// Cleanup
	for (int i = 0; i < N; i++)
		cout << a[i] << ", " << b[i] << ", " << c[i] << endl;
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
