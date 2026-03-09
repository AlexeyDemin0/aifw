#pragma once

#include <cstddef>

namespace aifw::core {

// from ../ops/kernel_registry.hpp
class IKernelRegistry;

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual void* allocate(size_t bytes) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void memcpy(void* dst, const void* src, size_t bytes) = 0;

  virtual IKernelRegistry& kernels() = 0;
};

}  // namespace aifw::core
