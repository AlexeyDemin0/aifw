#pragma once

#include <cstring>
#include <new>

#include "backend.hpp"

namespace aifw::core {

class CpuBackend final : public IBackend {
 public:
  void* allocate(size_t bytes) override;
  void deallocate(void* ptr) override;

  void memcpy(void* dst, const void* src, size_t bytes) override;
};

inline void* CpuBackend::allocate(size_t bytes) {
  return ::operator new(bytes, std::align_val_t(64));
}

inline void CpuBackend::deallocate(void* ptr) {
  ::operator delete(ptr, std::align_val_t(64));
}

inline void CpuBackend::memcpy(void* dst, const void* src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

}  // namespace aifw::core
