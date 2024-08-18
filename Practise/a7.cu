#include<iostream>
#include<string>
#include<cuda.h>

__global__ void RLECompression(const char* input, int inputSize, int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < inputSize) {
        char bit = input[tid];
        int count = 1;

        // Count consecutive occurrences of the current bit
        for (int i = tid + 1; i < inputSize; i++) {
            if (input[i] == bit) {
                count++;
            } else {
                if(input[i] != input[i-1]) output[tid] = count;
                break;
            }
        }

        // Write encoded data to output
    }
}

int main() {
    const char* input = "0001101000100011110111010001";
    int inputSize = strlen(input);

    char* d_input;
    int *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(char));
    cudaMalloc(&d_output, inputSize * sizeof(int));

    cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);

    // Define CUDA kernel launch configuration
    int blockSize = 256;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;

    // Launch CUDA kernel for parallel RLE compression
    RLECompression<<<numBlocks, blockSize>>>(d_input, inputSize, d_output);

    // Copy compressed data from device to host
    int* output = (int*)malloc(inputSize * sizeof(int));
    cudaMemcpy(output, d_output, inputSize * sizeof(int), cudaMemcpyDeviceToHost);
    char * str_output = (char *) malloc(10 * inputSize * sizeof(char));
    str_output[0] = input[0];
    int idx = 1;
    for(int i = 0; i < inputSize; i++){
        if(output[i] > 0){
            while(output[i] > 0){
                char num_ = '0' + output[i] % 10;
                output[i] /= 10;
                str_output[idx] = num_;
                idx ++;
            }
        }
    }
    // Print compressed data
    printf("Compressed data: %s\n", str_output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(output);

    return 0;
}
