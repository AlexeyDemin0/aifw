#pragma once

#include <cstddef>
#include <new>

#include "iallocator.hpp"

namespace aifw::core {

inline constexpr std::align_val_t kCpuAlignment{64};

class CpuAllocator final : public IAllocator {
 public:
  void* allocate(size_t bytes) override {
    return ::operator new(bytes, kCpuAlignment);
  }

  void deallocate(void* ptr) override { ::operator delete(ptr, kCpuAlignment); }
};

}  // namespace aifw::core
