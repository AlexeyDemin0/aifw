#pragma once

#include <cstddef>

namespace aifw::core {

class IAllocator {
 public:
  virtual ~IAllocator() = default;

  virtual void* allocate(size_t bytes) = 0;
  virtual void deallocate(void* prt) = 0;
};

}  // namespace aifw::core
