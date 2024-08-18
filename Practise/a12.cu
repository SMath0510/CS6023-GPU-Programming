#include <iostream>
#include <cuda.h>
#define N 1024
__global__ void dkernel(unsigned *a, unsigned a_size, unsigned chunksize) {
	// unsigned start = chunksize * threadIdx.x;
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	// for (unsigned nn = start; nn < start + chunksize; ++nn) {
	// 	a[nn]++;
	// }
    int n_threads = (blockDim.x * gridDim.x);
    for (unsigned ii = id; ii < a_size; ii += (n_threads)) {
		a[ii] ++;
	}
}
int main() {
	unsigned *a, chunksize = 32;
	cudaMalloc(&a, sizeof(unsigned) * N);
	dkernel<<<1, N/chunksize>>>(a, N, chunksize);
	cudaDeviceSynchronize();
    return 0;
}
