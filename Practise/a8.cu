#include<iostream>
#include<cuda.h>

__global__ 
void check(int * M, int N, int *flag){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N*N) return;
    int ii = id / N;
    int jj = id % N;
    if(ii == 0){
        int sum = 0;
        for(int col = 0; col < N; col ++){
            sum += M[ii*N + col];
        }
        if(sum != (N*(N+1))/2) atomicCAS(flag, 1, 0);
    }
    
    if(jj == 0){
        int sum = 0;
        for(int row = 0; row < N; row ++){
            sum += M[row*N + jj];
        }
        if(sum != (N*(N+1))/2) atomicCAS(flag, 1, 0);
    }

    if(ii%3 == 0 && jj%3 == 0){
        int sum = 0;
        for(int row = ii; row < ii + 3; row ++){
            for(int col = jj; col < jj + 3; col ++){
                sum += M[row*N + col];
            }
        }
        if(sum != (N*(N+1))/2) atomicCAS(flag, 1, 0);
    }
}

int main(){
    int N = 9;
    // int sudoku[81] = {
    //     5, 3, 0, 0, 7, 0, 0, 0, 0,
    //     6, 0, 0, 1, 9, 5, 0, 0, 0,
    //     0, 9, 8, 0, 0, 0, 0, 6, 0,
    //     8, 0, 0, 0, 6, 0, 0, 0, 3,
    //     4, 0, 0, 8, 0, 3, 0, 0, 1,
    //     7, 0, 0, 0, 2, 0, 0, 0, 6,
    //     0, 6, 0, 0, 0, 0, 2, 8, 0,
    //     0, 0, 0, 4, 1, 9, 0, 0, 5,
    //     0, 0, 0, 0, 8, 0, 0, 7, 0
    // };
    int sudoku[81] = {
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9
    };
    int * sudoku_g;
    cudaMalloc(&sudoku_g, N*N*sizeof(int));
    cudaMemcpy(sudoku_g, sudoku, N*N*sizeof(int), cudaMemcpyHostToDevice);
    int *flag = (int *) malloc(sizeof(int));
    *flag = 1;
    int * flag_g;
    cudaMalloc(&flag_g, sizeof(int));
    cudaMemcpy(flag_g, flag, sizeof(int), cudaMemcpyHostToDevice);
    check<<<1, 1024>>>(sudoku_g, N, flag_g);
    cudaMemcpy(flag, flag_g, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    printf("%d\n", *flag);
    
}