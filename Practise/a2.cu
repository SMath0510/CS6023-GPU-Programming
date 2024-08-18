#include<iostream>
#include<cuda.h>

__global__
void printer(int x){
    printf("Hello Kikks %d\n", x);
}

int main(){
    for(int i = 0; i < 100; i++){
        printer<<<1,1>>> (i); // executes sequentially
    }
    // printer<<<1,1>>>(1);
    // printer<<<1,1>>>(2);
    // printer<<<1,1>>>(3);
    cudaDeviceSynchronize();
}