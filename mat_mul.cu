#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matMulKernel(float *A, float *B, float *C, int M, int N,
                             int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < K) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
  }
}

int main() {
  int M = 1000; // Number of rows in A
  int N = 1000; // Number of columns in A and rows in B
  int K = 1000; // Number of columns in B

  // Allocate memory on host
  float *h_A = (float *)malloc(M * N * sizeof(float));
  float *h_B = (float *)malloc(N * K * sizeof(float));
  float *h_C = (float *)malloc(M * K * sizeof(float)); // result

  // Initialize input matrices
  for (int i = 0; i < M * N; i++) {
    h_A[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < N * K; i++) {
    h_B[i] = (float)rand() / RAND_MAX;
  }

  // Allocate memory on device
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_B, N * K * sizeof(float));
  cudaMalloc(&d_C, M * K * sizeof(float));

  // Copy input data from host to device
  cudaMemcpy(d_A, h_A, M * N * sizeof(float),
             cudaMemcpyHostToDevice); // void *dst
  cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

  // Set up execution configuration
  dim3 blockDim(16, 16);
  dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  // Launch kernel
  matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  // Copy result from device to host
  cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}