#include<iostream>
#include<cuda.h>
#include<fstream>

/* x2 + y3 */

__global__
void add(int *arr1, int *arr2, int N){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < N) arr1[id] += arr2[id];
}

int main(){
    int n;
    int * x_c, * y_c, * x_g, * y_g;
    std::ifstream x_inputs("x_inputs.txt");
    x_inputs >> n;
    x_c = (int *) malloc(n * sizeof(int));
    for(int i = 0; i < n; i++) x_inputs >> x_c[i];
    std::ifstream y_inputs("y_inputs.txt");
    y_inputs >> n;
    y_c = (int *) malloc(n * sizeof(int));
    for(int i = 0; i < n; i++) y_inputs >> y_c[i];
    cudaMalloc(&x_g, n*sizeof(int));
    cudaMalloc(&y_g, n*sizeof(int));
    cudaMemcpy(x_g, x_c, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y_g, y_c, n*sizeof(int), cudaMemcpyHostToDevice);
    int n_threads = 1024;
    int n_blocks = ceil((float)n /1024);
    add <<<n_blocks, n_threads>>>(x_g, y_g, n);
    cudaMemcpy(x_c, x_g, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++) std::cout << x_c[i] << " ";
    std::cout << std::endl;



    
    cudaDeviceSynchronize();
}

/*
Threads in a warp print in the threadIdx order
this is not necessary but some implementation of CUDA
*/