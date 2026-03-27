#include "../framework/bench.hpp"
#include "aifw/core/device/device_registry.hpp"
#include "aifw/core/ops/ops.hpp"

using namespace aifw::core;

BENCH(Matmul, 64x64_float32, 200) {
  static Tensor a(cpu(), Shape{64, 64}, DType::Float32);
  static Tensor b(cpu(), Shape{64, 64}, DType::Float32);
  static Tensor out(cpu(), Shape{64, 64}, DType::Float32);
  ops::matmul(a, b, out);
}

BENCH(Matmul, 256x256_float32, 50) {
  static Tensor a(cpu(), Shape{256, 256}, DType::Float32);
  static Tensor b(cpu(), Shape{256, 256}, DType::Float32);
  static Tensor out(cpu(), Shape{256, 256}, DType::Float32);
  ops::matmul(a, b, out);
}

BENCH(Matmul, 512x512_float32, 10) {
  static Tensor a(cpu(), Shape{512, 512}, DType::Float32);
  static Tensor b(cpu(), Shape{512, 512}, DType::Float32);
  static Tensor out(cpu(), Shape{512, 512}, DType::Float32);
  ops::matmul(a, b, out);
}

BENCH(Matmul, 256x256_float64, 50) {
  static Tensor a(cpu(), Shape{256, 256}, DType::Float64);
  static Tensor b(cpu(), Shape{256, 256}, DType::Float64);
  static Tensor out(cpu(), Shape{256, 256}, DType::Float64);
  ops::matmul(a, b, out);
}
