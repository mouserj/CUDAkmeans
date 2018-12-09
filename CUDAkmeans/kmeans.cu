/***
* @file kmeans.cu
* @authors: Jake Mouser & John Nguyen
*
* A Parallel approach to Lloyd's algorithm/K-Means Clustering using CUDA
* Requires an Nvidia graphics card and appropriate CUDA toolkit and drivers
* Currently uses randomly generated data to demonstrate the speedup of the parallel kmeans algorithm
*/
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

/***
* eucdistance() - takes a point and centroid and returns the square of the euclidean distance
* @param datum double* points to a single data point
* @param cent double* points to a single centroid
* @param dim int the dimensionality of the input data
* We return the square of the euclidean distance as this is only used in comparison to other distances,
* for two distances a,b: a < b	implies sqrt(a) <  sqrt(b)
*/
static inline __device__ double eucDistance(double* datum, double* cent, int dim) {
	double distance = 0.0;
	for (int i = 0; i < dim; i++) {
		//sum the difference between each dimension
		distance += (datum[i] - cent[i]) * (datum[i] - cent[i]); //squared values will always be positive
	}
	return distance;
}

/***
* labelNearest() - global function for GPU to label data points in parallel
* @param data double* points to an array of n data points
* @param centroids double* points to an array of k centroids
* @param out int points to the array for the output labels
* @param n int number of data points
* @param k int number of centroids
* @param dim int the dimensionality of the input data
* This function is called by each thread within the GPU. This takes a point, and compares each centroid to find the closest centroid
* Modifies the output array of the corresponding index to write out the label
*/
__global__ void labelNearest(double* data, double* centroids, int* out, int n, int k, int dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		out[index] = 0;
		double min_distance = eucDistance(data + index * dim, centroids, dim);//check distance on each cluster to find minimum
		for (int i = 1; i < k; i++) {
			double this_Distance = eucDistance(data + index * dim, centroids + i * dim, dim);
			if (min_distance > this_Distance) {
				out[index] = i;
				min_distance = this_Distance;
			}
		}
	}
}

/***
* fRand() - takes a range and generates a random double within the range.
* @param min double minimum value for the random output
* @param max double maximum value for the random output
* This function is used for data generation, as well as generation of the centroids.
*/
static double fRand(double min, double max) {
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}

/***
* Kmeans Class 
*
* 
*/
class Kmeans {
public:
	//Constructor
	Kmeans(double* data, int n, int dim) {
		this->data = data;
		this->n = n;
		this->dim = dim;
		this->k = 3;
		this->maxIter = 1000;
	}

	/***
	* cluster()
	* @param k integer number of cluster
	* @param maxIter in maximum number of iterations before stopping
	* Once the Kmeans object is initialized this function initializes the GPU and processes the data to create cluster labels
	*/
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

	/***
	* initializeCuda()
	* Allocates memory on the GPU and sets up cuda arrays
	* Copies the data from the current object from main memory to the GPU memory
	*/
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

	/***
	* closeCuda()
	* Frees GPU allocated memory
	*/
	void closeCuda() {
		cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_out);
	}

	/***
	* nearestCentroids()
	* copies the array of centroids to the GPU then starts the kernel function on the GPU
	* This function then blocks until the kernel has completed on all threads, and copies the output back to the host machine
	*/
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

	/***
	* calcCentroids()
	* Uses the output labels to caclulate new centroids,
	* This function is currently sequential but could possibly be imporved to be processed in parallel
	*/
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

	/***
	* randCentroid()
	* Creates the initial centroids using random number generation
	* This processes the input set of data to ensure that the centroids are contained within the min and max of each field
	*/
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

	// class fields
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
	//prevents terminal from closing after completions until keypress
	char blah;
	cin >> blah;
	//free memory
	delete[] data;
	delete[] out;
	return 0;
}
