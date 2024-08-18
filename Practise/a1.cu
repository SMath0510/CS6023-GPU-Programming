#include<iostream>
#include<cuda.h>

__global__
void increment(int * x){
    atomicAdd(x, 1);
    // x[0] += 1;
}

int main(){
    int * x_g, * x_c;
    x_c = (int *) malloc(sizeof(int));
    cudaMalloc(&x_g, sizeof(int));
    // increment<<<1,100>>> (x_g); -> 1
    increment<<<100,1>>> (x_g); // -> 1
    cudaMemcpy(x_c, x_g, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", x_c[0]);

}