
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <vector>
//#include <float.h>


using namespace std;


#define N 256 //data size
#define M 256 //threads per block

__global__ void add(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
		c[index] = a[index] + b[index];
}

static inline __device__ double eucdistance(double* data, double* centroid, int dim) {
	double distance = 0.0;
	int size = sizeof(double); //can take 
	for (int i = 0; i < dim; i += size) {
		//take the square of the difference and add it to a running sum
		distance += (data[i] - centroid[i]) * (data[i] - centroid[i]); //squared values will always be positive
	}
	//could take the sqrt but if a < b	implies sqrt(a) <  sqrt(b)
	return distance;
}


__global__ void labelNearest(double* data, double* centroids, int* out, int n, int k, int dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index % dim == 0) {	//check to see if its start of point

		out[index / 3] = 0;
		double min_distance = eucdistance(data + index, centroids, dim);//check distance on each cluster to find minimum
		for (int i = 1; i < k; i++) {
			double this_Distance = eucdistance(data + index, centroids + i, dim);
			if (min_distance > this_Distance) {
				out[index / 3] = i;
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
			converged = calcCentroids(out);	//make this parallel
			iter++;
		}
		closeCuda();
		return out;
	}


private:

	/*sets up cuda arrays*/
	void initializeCuda() {
		//data sizes
		int data_size = n * dim * sizeof(double);
		int cent_size = n * k * sizeof(double);
		int out_size = n * sizeof(int);

		// Alloc space for device copies of a, b, c
		cudaMalloc((void **)&d_data, data_size);
		cudaMalloc((void **)&d_centroids, cent_size);
		cudaMalloc((void **)&d_out, out_size);

		// Alloc space for host copy of output
		out = (int *)malloc(out_size);

		// Copy inputs to device
		cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
	}

	/*frees device memory*/
	void closeCuda() {
		cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_out);
	}

	/* finds nearest centroids, must make parallel*/
	void nearestCentroids(){

		//copy centroid data to GPU
		int cent_size = n * k * sizeof(double);
		cudaMemcpy(d_centroids, centroids, cent_size, cudaMemcpyHostToDevice);

		// Launch kernel on GPU
		labelNearest<<< (n + M - 1) / M, M >>> (d_data, d_centroids, d_out, int n, int k, int dim);

		// Copy result back to host
		int out_size = n * sizeof(int);
		cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);


	}

	/*recaclulates new centroids, must make parallel*/
	bool calcCentroids(int *out) {
		bool converged = true;
		double* old = centroids; //for determining convergence
		centroids = new double[k*dim];
		int* count = new int[k];

		//add up all of the data to the respective centers
		for (int i = 0; i < n; i++) {
			int current = out[i];
			for (int j = 0; j < dim; j++) {
				centroids[current*dim + j] += data[i*dim + j];
				count[current] += 1;
			}
		}

		//average the data and test for convergence
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dim; j++) {
				centroids[i*dim + j] /= count[i];
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
		vector<double> min = { DBL_MAX, DBL_MAX, DBL_MAX };
		vector<double> max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
	for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim; j++) {
				if (data[i*dim + j] > max.at(j))
					max.at(j) = data[i*dim + j];
				if (data[i*dim + j] < min.at(j))
					min.at(j) = data[i*dim + j];
			}
		}
		centroids = new double[k*dim];
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dim; j++) {
				centroids[i*dim + j] = fRand(min.at(k), max.at(k));
			}
		}
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
	static const int MAXITER = 1000;
	static const int K = 3;
	static const int DIM = 2;
	static const double DATAMAX = 10.0;
	static const double DATAMIN = -10.0;

	//generate test data
	double *data = new double[N*DIM];
	for (int i = 0; i < N*DIM; i++) {
		data[i] = fRand(DATAMIN, DATAMAX);
	}

	//initialize kmeans class
	Kmeans test = Kmeans(data, N, DIM);

	//run kmeans
	int* out = test.cluster(K, MAXITER);

	//print results
	for (int i = 0; i < N; i++) {
		cout << out[i] << endl;
	}

	//free memory
	delete[] data;
	delete[] out;
	return 0;
}
