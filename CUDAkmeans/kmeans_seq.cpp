/***
* @file kmeans.cu
* @authors: Jake Mouser & John Nguyen
*
* A sequential approach to Lloyd's algorithm/K-Means Clustering using CUDA
* Uses the exact same process as kmeans.cu from this project but does not use the GPU to calculate labels in parallel
*/
#include <iostream>
#include <vector>
#include <float.h>
#include <chrono>


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
static inline double eucdistance(double* datum, double* cent, int dim) {
	double distance = 0.0;
	for (int i = 0; i < dim; i++) {
		//take the square of the difference and add it to a running sum
		distance += (datum[i] - cent[i]) * (datum[i] - cent[i]); //squared values will always be positive
	}
	//could take the sqrt but if a < b	implies sqrt(a) <  sqrt(b)
	return distance;
}

/***
* labelNearest() - labels a data point to its nearest centroid
* @param data double* points to an array of n data points
* @param centroids double* points to an array of k centroids
* @param out int points to the array for the output labels
* @param n int number of data points
* @param k int number of centroids
* @param dim int the dimensionality of the input data
* This takes a point, and compares each centroid to find the closest centroid
* Modifies the output array of the corresponding index to write out the label
*/
void labelNearest(double* data, double* centroids, int* out, int n, int k, int dim) {
	for (int index = 0; index < n; index++) {
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
* KmeansSequential Class
*
*
*/
class KmeansSequential {
public:
	KmeansSequential(double* data, int n, int dim) {
		this->data = data;
		this->n = n;
		this->dim = dim;
		this->k = 3;
		this->maxIter = 1000;
		this->out = new int[n];
	}

	/***
	* cluster()
	* @param k integer number of cluster
	* @param maxIter in maximum number of iterations before stopping
	* Once the Kmeans object is initialized this function runs the main loop for the program
	*/
	int* cluster(int k, int maxIter) {
		this->k = k;
		this->maxIter = maxIter;
		randCentroids();	//make this parallel
		int iter = 0;
		bool converged = false;
		while (!converged && iter < maxIter) {
			labelNearest(data, centroids, out, n, k, dim);	//make this parallel

			converged = calcCentroids();	//make this parallel

			iter++;
			cout << "num Iterations: " << iter << endl;
		}
		return out;
	}


private:

	/***
	* calcCentroids()
	* Uses the output labels to caclulate new centroids
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

	// fields
	double* data;
	double* centroids;
	int* out;

	int n;
	int dim;
	int k;
	int maxIter;

};

int main(void) {
	//define constants
	static const int MAXITER = 20;
	static const int K = 4;
	static const int DIM = 4;
	static const double DATAMAX = 10.0;
	static const double DATAMIN = -10.0;

	//generate test data
	cout << "Generating test data" << endl;
	double *data = new double[N*DIM];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < DIM; j++) {
			data[i*DIM + j] = fRand(DATAMIN, DATAMAX);
		}
	}
	auto start = chrono::steady_clock::now();
	//initialize kmeans class
	KmeansSequential test = KmeansSequential(data, N, DIM);

	//run kmeans
	int* out = test.cluster(K, MAXITER);
	auto end = chrono::steady_clock::now(); auto elpased = chrono::duration<double, milli>(end - start).count();
	cout << "Processed " << MAXITER << " in " << elpased << "ms" << endl;
	//print first 16 results
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
