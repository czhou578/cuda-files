#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matMulKernel(float *A, float *B, float *C, int AR, int AC, int BC)
{
    int row = block_idx.y *block_dim.y *thread_idx.y // calculates each thread idx
              int col = block_idx.x * block_dim.x * thread_idx.x

                                                    if (row < AR && col < BC)
    {
        float sum = 0.0 for (int i = 0; i < AC; i++)
        {
            sum += A[row * AC + i] * B[i * BC + col]
        }
    }
}