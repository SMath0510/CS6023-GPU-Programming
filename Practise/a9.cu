#include <iostream>
#include <cuda.h>

// Define a struct representing a 2D point
struct Point {
    float x;
    float y;
};

// Kernel function to modify the coordinates of a point
__global__ void modifyPoint(Point* point) {
    // Modify the coordinates of the point in the kernel
    point->x *= 2;
    point->y *= 2;
}

int main() {
    // Declare and initialize a point on the host (CPU)
    Point hostPoint = {1.0f, 2.0f};

    // Allocate memory for the point on the GPU
    Point* devicePoint;
    cudaMalloc(&devicePoint, sizeof(Point));

    // Copy the point from host to device
    cudaMemcpy(devicePoint, &hostPoint, sizeof(Point), cudaMemcpyHostToDevice);

    // Launch the kernel to modify the coordinates of the point
    modifyPoint<<<1, 1>>>(devicePoint);
    cudaDeviceSynchronize();

    // Copy the modified point back from device to host
    cudaMemcpy(&hostPoint, devicePoint, sizeof(Point), cudaMemcpyDeviceToHost);

    // Print the modified point
    printf("Modified Point: (%.2f, %.2f)\n", hostPoint.x, hostPoint.y);

    // Free memory on the device
    cudaFree(devicePoint);

    return 0;
}
