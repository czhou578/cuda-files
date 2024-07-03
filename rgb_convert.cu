#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t result = call;                                                 \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", __FILE__,   \
              __LINE__, static_cast<unsigned int>(result),                     \
              cudaGetErrorString(result), #call);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__global__ void rgba_grey(const unsigned char *input, unsigned char *output,
                          int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    int rgb_idx = idx * 3;
    output[idx] = static_cast<unsigned char>(0.2989f * input[rgb_idx] +
                                             0.5870f * input[rgb_idx + 1] +
                                             0.1140f * input[rgb_idx + 2]);
  }
}

void rgb_to_grayscale(const unsigned char *h_input, unsigned char *h_output,
                      int width, int height) {
  size_t rgb_size = width * height * 3 * sizeof(unsigned char);
  size_t gray_size = width * height * sizeof(unsigned char);

  // Allocate device memory
  unsigned char *d_input, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, rgb_size));
  CHECK_CUDA(cudaMalloc(&d_output, gray_size));

  // Copy input from host to device
  CHECK_CUDA(cudaMemcpy(d_input, h_input, rgb_size, cudaMemcpyHostToDevice));

  // Set up grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim(cdiv(width, blockDim.x), cdiv(height, blockDim.y));

  // Launch kernel
  rgba_grey<<<gridDim, blockDim>>>(d_input, d_output, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result from device to host
  CHECK_CUDA(cudaMemcpy(h_output, d_output, gray_size, cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
}

int main() {
  const int width = 1920;
  const int height = 1080;

  // Allocate host memory
  unsigned char *h_input = new unsigned char[width * height * 3];
  unsigned char *h_output = new unsigned char[width * height];

  // TODO: Fill h_input with actual RGB image data

  // Convert RGB to grayscale
  rgb_to_grayscale(h_input, h_output, width, height);

  // TODO: Save or process h_output as needed

  // Free host memory
  delete[] h_input;
  delete[] h_output;

  return 0;
}