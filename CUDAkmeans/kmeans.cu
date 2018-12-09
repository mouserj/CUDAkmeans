
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <float.h>


using namespace std;


#define N 33554432 //data size
#define M 512 //threads per block


static inline __device__ double eucdistance(double* datum, double* cent, int dim) {
	double distance = 0.0;
	for (int i = 0; i < dim; i++) {
		//take the square of the difference and add it to a running sum
		distance += (datum[i] - cent[i]) * (datum[i] - cent[i]); //squared values will always be positive
	}
	//could take the sqrt but if a < b	implies sqrt(a) <  sqrt(b)
	return distance;
}


__global__ void labelNearest(double* data, double* centroids, int* out, int n, int k, int dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		out[index] = 0;
		double min_distance = eucdistance(data + index * dim, centroids, dim);//check distance on each cluster to find minimum
		for (int i = 1; i < k; i++) {
			double this_Distance = eucdistance(data + index * dim, centroids + i * dim, dim);
			if (min_distance > this_Distance) {
				out[index] = i;
				min_distance = this_Distance;
			}
		}
	}
}

void sequential_ints(int* a, int size)
{
	for (int i = 0; i < size; i++)
		a[i] = i;
}

static double fRand(double min, double max) {
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}

class Kmeans {
public:
	Kmeans(double* data, int n, int dim) {
		this->data = data;
		this->n = n;
		this->dim = dim;
		this->k = 3;
		this->maxIter = 1000;
	}


	int* cluster(int k, int maxIter) {
		this->k = k;
		this->maxIter = maxIter;
		randCentroids();	//make this parallel
		int iter = 0;
		bool converged = false;
		initializeCuda();
		while (!converged && iter < maxIter) {
			nearestCentroids();	//make this parallel

			converged = calcCentroids();	//make this parallel


			iter++;
			cout << "num Iterations: " << iter << endl;
		}
		closeCuda();
		return out;
	}


private:

	/*sets up cuda arrays*/
	void initializeCuda() {
		//data sizes
		int data_size = n * dim * sizeof(double);
		int cent_size = k * dim * sizeof(double);
		int out_size = n * sizeof(int);

		// Alloc space for device copies of a, b, c
		cudaMalloc((void **)&d_data, data_size);
		cudaMalloc((void **)&d_centroids, cent_size);
		cudaMalloc((void **)&d_out, out_size);

		// Alloc space for host copy of output
		out = new int[n];
		for (int i = 0; i < n; i++)
			out[i] = 0;

		// Copy inputs to device
		cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_out, out, out_size, cudaMemcpyHostToDevice);
	}

	/*frees device memory*/
	void closeCuda() {
		cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_out);
	}

	/* finds nearest centroids, must make parallel*/
	void nearestCentroids() {

		//copy centroid data to GPU
		int cent_size = dim * k * sizeof(double);
		cudaMemcpy(d_centroids, centroids, cent_size, cudaMemcpyHostToDevice);

		// Launch kernel on GPU
		labelNearest << < (n + M - 1) / M, M >> > (d_data, d_centroids, d_out, n, k, dim);
		cudaDeviceSynchronize();
		// Copy result back to host
		int out_size = n * sizeof(int);
		cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);


	}

	/*recaclulates new centroids, must make parallel*/
	bool calcCentroids() {
		bool converged = true;
		double* old = centroids; //for determining convergence

		//initialize new centroids and count array
		int* count = new int[k];
		centroids = new double[k*dim];
		for (int i = 0; i < k; i++) {
			count[i] = 0;
			for (int j = 0; j < dim; j++)
				centroids[i*dim + j] = 0;
		}

		//add up all of the data to the respective centers
		for (int i = 0; i < n; i++) {
			int current = out[i];
			for (int j = 0; j < dim; j++) {
				centroids[current*dim + j] += data[i*dim + j];
			}
			count[current] += 1;
		}


		//average the data and test for convergence
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dim; j++) {
				centroids[i*dim + j] /= (double)count[i];
				if (centroids[i*dim + j] != old[i*dim + j])
					converged = false;
			}
		}
		delete[] old;
		delete[] count;
		return converged;
	}

	/** I think we can paralellize this*/
	void randCentroids() {
		//get range of dataset
		double* min = new double[dim];
		double* max = new double[dim];
		for (int i = 0; i < dim; i++) {
			min[i] = DBL_MAX;
			max[i] = -DBL_MAX;
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim; j++) {
				if (data[i*dim + j] > max[j])
					max[j] = data[i*dim + j];
				if (data[i*dim + j] < min[j])
					min[j] = data[i*dim + j];
			}
		}
		centroids = new double[k*dim];
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dim; j++) {
				centroids[i*dim + j] = fRand(min[j], max[j]);
			}
		}
		delete[] min;
		delete[] max;
	}

	// fields
	double* data;
	double* centroids;
	int* out;

	// device copies of data, centroids, output
	double* d_data;
	double* d_centroids;
	int* d_out;

	int n;
	int dim;
	int k;
	int maxIter;

};

int main(void) {
	//define constants
	static const int MAXITER = 100;
	static const int K = 4;
	static const int DIM = 4;
	static const double DATAMAX = 10.0;
	static const double DATAMIN = -10.0;
	cout << "Generating test data" << endl;
	//generate test data
	double *data = new double[N*DIM];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < DIM; j++) {
			data[i*DIM + j] = fRand(DATAMIN, DATAMAX);
		}
	}


	auto start = chrono::steady_clock::now();
	//initialize kmeans class
	Kmeans test = Kmeans(data, N, DIM);

	//run kmeans
	int* out = test.cluster(K, MAXITER);
	auto end = chrono::steady_clock::now(); auto elpased = chrono::duration<double, milli>(end - start).count();
	cout << "Processed " << MAXITER << " in " << elpased << "ms" << endl;

	
	//print results
	cout << "First 16 results: " << endl;
	for (int i = 0; i < 16; i++) {
		cout << out[i] << endl;

	}
	char blah;
	cin >> blah;
	//free memory
	delete[] data;
	delete[] out;
	return 0;
}
