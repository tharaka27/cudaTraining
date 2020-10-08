#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <ctime>
#include <chrono> 

using namespace std;

__global__ void AddMat(int* a, int* b, int count) {
 
	//int id = blockIdx.x * blockDim.x + threadIdx.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < count) {
		a[index] += b[index];
	}

}

int main() {

	int count = 30000000;
	int *h_a = new int[count];
	int *h_b = new int[count];
	int* h_c = new int[count];

	srand(time(NULL));
 

	for (int i = 0; i < count; i++) {
		h_a[i] = rand() % 1000;
		h_b[i] = rand() % 1000;
	
	}

	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < count; i++) {
		h_c[i] = h_a[i] + h_b[i];
	}
	auto end = chrono::high_resolution_clock::now();
	double time_taken_CPU = chrono::duration_cast<chrono::nanoseconds>(end - start).count();




	for (int i = 0; i < 5; i++) {
		cout << h_a[i] << " + " << h_b[i] << endl;

	}

	int *d_a, *d_b;

	if (cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess) {
		cout << "Cound not allocate enough memory to d_a variable" << endl;
		return 0;
	}

	if (cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess) {
		cout << "Cound not allocate enough memory to d_a variable" << endl;
		cudaFree(d_a);
		return 0;
	}


	if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Cound not copy memory to d_a variable" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 0;
	}

	if (cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Cound not copy memory to d_b variable" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 0;
	}
	
	start = chrono::high_resolution_clock::now();
	
	AddMat <<<count/256 + 1,256>>>(d_a, d_b, count);

	end = chrono::high_resolution_clock::now();
	double time_taken_GPU = chrono::duration_cast<chrono::nanoseconds>(end - start).count();


	if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Cound not copy memory to d_a variable" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 0;
	}

	cudaFree(d_a);
	cudaFree(d_b);
	
	
	for (int i = 0; i < 5; i++) {
		cout << h_a[i] << endl;

	}

	cout << "Calculate value difference between CPU and GPU calculation" << endl;
	int notchange = 0;
	for (int i = 0; i < count; i++) {
		if (h_a[i] == h_c[i]) {
			notchange++;
		}
	}
	time_taken_CPU *= 1e-9;
	time_taken_GPU *= 1e-9;
	cout << "equal/total: " << notchange <<"/" << count << endl;
	cout << "CPU time: " << time_taken_CPU  <<"s" << endl;
	cout << "GPU time: " << time_taken_GPU  <<"s" << endl;
	cout << "Speed UP in GPU: " << time_taken_CPU/time_taken_GPU << "x times CPU"  << endl;

	free(h_a);
	free(h_b);
	return 0;
}