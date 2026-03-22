#pragma once

#include "aifw/core/device/device.hpp"
#include "aifw/core/device/iallocator.hpp"
#include "aifw/core/device/ikernel_registry.hpp"

namespace aifw::core {

class IDevice {
 public:
  virtual ~IDevice() = default;

  virtual Device device() const = 0;
  virtual IAllocator& allocator() = 0;
  virtual IKernelRegistry& kernels() = 0;
};

}  // namespace aifw::core
