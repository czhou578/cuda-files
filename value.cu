#include <cuda_runtime.h>
#include <functional>
#include <set>

#ifndef Value
#define Value

class Value {
public:
  float grad;
  std::set<Value *> prev;
  std::string op;
  std::function<void()> backward;

  __host__ __device__ Value(float data,
                            const std::vector<Value *> &children = {}, const)

      __device__ void backward() {
    backward();
  }
}

__global__ void
createValue(Value *v, float data) {
  *v = Value(data);
}
