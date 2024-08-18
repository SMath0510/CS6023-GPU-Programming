#include<iostream>
#include<cuda.h>

__global__
void dkernel(){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < 32; i++){
        if(id % 32 == i && (id/32) == 0){
            printf("%d\n", id * 2);
        }
        __syncthreads();
        if(id % 32 == i && (id/32) == 1){
            printf("%d\n",(id-32) * 2 + 1);
        }
        __syncthreads();
    }
}

int main(){
    dkernel<<<1, 64>>>();
    cudaDeviceSynchronize();
}