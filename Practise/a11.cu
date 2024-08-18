#include<iostream>
#include<cuda.h>

#define N 1024

__device__ int flag[N];
__device__ int label[N];

/* Used for functions and global variables that are called within the GPU */
__device__ 
__host__
int add(int a, int b){
    return a + b;
}

__global__
void dkernel(){
    printf("Just checking %d\n", add(1, 2));
}

int main(){
    dkernel<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << add(1, 2) << std::endl;
    return EXIT_SUCCESS;
}