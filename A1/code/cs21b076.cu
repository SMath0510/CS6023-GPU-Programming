/**
*   CS6023: GPU Programming
*   Assignment 1
*
*   Please don't change any existing code in this file.
*
*   You can add your code whereever needed. Please add necessary memory APIs
*   for your implementation. Use cudaFree() to free up memory as soon as you're
*   done with an allocation. This will ensure that you don't run out of memory
*   while running large test cases. Use the minimum required memory for your
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>

using std::cin;
using std::cout;

#define __DEBUG__ 0
#define __PRINT_MATRIX_ 0


__global__
void CalculateHadamardProduct(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    unsigned id = (blockDim.x)*(blockIdx.x) + threadIdx.x;
    unsigned ii = id / N;
    unsigned jj = id % N;
    unsigned b_id = jj * N + ii; // B_transposed
    unsigned a_id = ii * N + jj;
    unsigned max_size = N*N;
    if(id < max_size){
        A[a_id] = A[a_id] * B[b_id];
    }
}

__global__
void FindWeightMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    /*
        BLOCK (blockDim.x * blockDim.y threads) -> THREAD_X(blockDim.y threads) -> THREAD_Y
    */
    unsigned id = (blockDim.x * blockDim.y) * blockIdx.x + (blockDim.y) * threadIdx.x + threadIdx.y;
    unsigned max_size = N*N;
    if(id < max_size){
        long long int max_ele = A[id];
        if(B[id] > A[id]){
            max_ele = B[id];
        }
        A[id] = max_ele;
    }
}

__global__
void CalculateFinalMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    unsigned B_id = ((blockIdx.x * gridDim.y) + blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
    unsigned B_ii = B_id / (2*N);
    unsigned B_jj = B_id % (2*N);

    unsigned A_ii = B_ii % N;
    unsigned A_jj = B_jj % N;
    unsigned A_id = A_ii * N + A_jj;

    unsigned A_max_size = N*N;
    unsigned B_max_size = (2*N)*(2*N);
    if(A_id < A_max_size && B_id < B_max_size){
        B[B_id] = A[A_id] * B[B_id];
    }
}


int main(int argc, char** argv) {


    int N;
    cin >> N;
    long int* A = new long int[N * N];
    long int* B = new long int[N * N];
    long int* C = new long int[N * N];
    long int* D = new long int[2 * N * 2 * N];


    for (long int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> C[i];
    }

    for (long int i = 0; i < 2 * N * 2 * N; i++) {
        cin >> D[i];
    }

    /*
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    */

    long int* d_A;
    long int* d_B;
    long int* d_C;
    long int* d_D;

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(N * N / 1024.0), 1, 1);

    unsigned size1 = N*N*sizeof(long long int);
    unsigned size2 = (2*N)*(2*N)*sizeof(long long int);

    cudaMalloc(&d_A, size1);
    cudaMalloc(&d_B, size1);
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size1, cudaMemcpyHostToDevice);

    CalculateHadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, size1, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    cudaFree(d_B);

    if(__DEBUG__) cout << "Hadamard Product Calculated\n";

    if(__PRINT_MATRIX_){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                cout << A[i*N + j] << " ";
            }
            cout << "\n";
        }
    }

    
    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(N * N / 1024.0), 1, 1);

    cudaMalloc(&d_C, size1);
    start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size1, cudaMemcpyHostToDevice);

    FindWeightMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, size1, cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    cudaFree(d_C);

    if(__DEBUG__) cout << "Weight Matrix Calculated\n";
    if(__PRINT_MATRIX_){
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                cout << A[i*N + j] << " ";
            }
            cout << "\n";
        }
    }

    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);

    cudaMalloc(&d_D, size2);
    start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, size2, cudaMemcpyHostToDevice);

    CalculateFinalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_D, N);

    cudaDeviceSynchronize();
    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_D);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end - start;
    
    if(__DEBUG__) cout << "Final Matrix Calculated\n";
    if(__PRINT_MATRIX_){
        for(int i = 0; i < 2*N; i++){
            for(int j = 0; j < 2*N; j++){
                cout << D[i*2*N + j] << " ";
            }
            cout << "\n";
        }
    }

    // Make sure your final output from the device is stored in d_D.

    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */

    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_D);
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < 2 * N; i++) {
            for (long int j = 0; j < 2 * N; j++) {
                file << D[i * 2 * N + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2 << elapsed2.count() << "\n";
        file2 << elapsed3.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}