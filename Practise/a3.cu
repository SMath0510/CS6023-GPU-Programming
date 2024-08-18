#include<iostream>
#include<cuda.h>

__global__
void printer(){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("Hello Kikks %d\n", id);
}

int main(){
    printer<<<10,10>>>();
    cudaDeviceSynchronize();
}

/*
Threads in a warp print in the threadIdx order
this is not necessary but some implementation of CUDA
*/