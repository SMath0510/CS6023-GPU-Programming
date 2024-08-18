#include<stdio.h>
#include<cuda.h>

__device__ int shaun;

__host__ __device__ 
void fun(int * count){printf("%d\n", *count);} 

__global__
void check(int * ccount, int *gcount){
    fun(ccount);  // prints nothing, just empty's buffer
    *gcount = 100; 
}

int main(){
    int * gcount;
    cudaMalloc(&gcount, sizeof(int));
    int * ccount = (int *)malloc(sizeof(int));
    cudaMemset(gcount, 0, sizeof(int));
    memset(ccount, 0, sizeof(int));

    *ccount = 10;
    check<<<1,1>>> (ccount, gcount); 
    cudaDeviceSynchronize();
    // fun(gcount); // Gives seg fault
}