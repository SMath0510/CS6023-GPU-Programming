#include<iostream>
#include<cuda.h>
#include<fstream>

/* x2 + y2 */

#define DEBUG 0

__global__
void solver(int *arr1, int *mx, int N){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id % 2 || id >= 2*N) return;
    int dist = arr1[id]*arr1[id] + arr1[id+1]*arr1[id+1];
    if(DEBUG) printf("%d\n", dist);
    atomicMax(mx, dist);
}

int main(){
    int n;
    int * x_c, * y_c, * x_g, * y_g;
    std::ifstream coord_inputs("coords_input.txt");
    coord_inputs >> n;
    x_c = (int *) malloc(2*n * sizeof(int));
    for(int i = 0; i < 2*n; i+=2) coord_inputs >> x_c[i] >> x_c[i+1];
    
    if(DEBUG) for (int i = 0; i < 2*n; i+=2) std::cout << x_c[i] << " " << x_c[i+1] << std::endl;
    cudaMalloc(&x_g, 2*n*sizeof(int));
    // cudaMalloc(&y_g, n*sizeof(int));
    cudaMemcpy(x_g, x_c, 2*n*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(y_g, y_c, n*sizeof(int), cudaMemcpyHostToDevice);
    int n_threads = 1024;
    int n_blocks = ceil((float)2*n /1024);
    n_blocks = 1;
    int *max_dist_g, *max_dist_c;
    max_dist_c = (int *)malloc(sizeof(int));
    cudaMalloc(&max_dist_g, sizeof(int));
    solver <<<n_blocks, n_threads>>>(x_g, max_dist_g, n);
    cudaMemcpy(max_dist_c, max_dist_g, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << *max_dist_c << std::endl;



    
}

/*
Threads in a warp print in the threadIdx order
this is not necessary but some implementation of CUDA
*/