#include<stdio.h>
#include<cuda.h>

__device__ int shaun;

__host__ __device__ 
void fun(){shaun ++;} 

__global__
void check(){
    fun();
    printf("%d\n", shaun);
}

int main(){
    fun(); // useless call
    fun(); // useless call - cannot access the __device__ variable
    check<<<1,1>>>();
    cudaDeviceSynchronize();
}