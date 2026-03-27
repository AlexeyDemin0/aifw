#pragma once

#include <cstddef>

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

  virtual void read_bytes(
      void* dst_host, const void* src_device, size_t bytes
  ) = 0;
  virtual void write_bytes(
      void* dst_device, const void* src_host, size_t bytes
  ) = 0;
};

}  // namespace aifw::core
