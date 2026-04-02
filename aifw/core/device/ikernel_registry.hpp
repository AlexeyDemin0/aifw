#pragma once

#include <cstddef>
#include <vector>

namespace aifw::core {

class Tensor;

class IKernelRegistry {
 public:
  virtual ~IKernelRegistry() = default;

  virtual void fill(Tensor& t, double value) = 0;
  virtual void add(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual void sub(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual void mul(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual void div(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual void matmul(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual void relu(const Tensor& a, Tensor& out) = 0;

  virtual void fill_diagonal(Tensor& t, double value) = 0;
  virtual void arange(Tensor& t, double start, double step) = 0;

  virtual void sum(
      const Tensor& input,
      Tensor& out,
      const std::vector<size_t>& axes,
      bool keepdims
  ) = 0;
};

}  // namespace aifw::core
