#include<stdio.h>
#include<cuda.h>

__host__ __device__ 
void fun(int * count){
    *count = 100;
} 

__global__
void check(int * count){
    fun(count);  // error call 
}

int main(){
    int * count;
    int * ccount = (int *) malloc(sizeof(int));
    cudaMalloc(&count, sizeof(int));
    cudaMemset(count, 0, sizeof(int));
    check<<<1,1>>> (count);
    check<<<1,1>>> (ccount); // doesnt do anything, illegal memory access
    cudaDeviceSynchronize();
    cudaMemcpy(ccount, count, sizeof(int), cudaMemcpyDeviceToHost);
    /*
        fun(count); // leads to segmentation fault as count cannot be accessed from cpu
    */
    printf("%d\n", *ccount);

}