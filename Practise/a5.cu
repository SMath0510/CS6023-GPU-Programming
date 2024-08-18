#include<iostream>
#include<cuda.h>

__device__ int i = 0;
int dd = 0;

__global__
void dkernel(){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < 64) atomicAdd(&i, 1);

    __syncthreads();
    printf("%d\n", i);
    printf("%d\n", dd);
}

int main(){
    dkernel<<<2, 33>>> ();
    cudaDeviceSynchronize();
    std:: cout << i << std:: endl;
}

/*
Threads in a warp print in the threadIdx order
this is not necessary but some implementation of CUDA
*/