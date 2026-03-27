#pragma once

#include <cstring>

#include "aifw/core/device/iallocator.hpp"
#include "aifw/core/device/idevice.hpp"
#include "aifw/core/device/ikernel_registry.hpp"
#include "cpu_allocator.hpp"
#include "cpu_kernel_registry.hpp"

namespace aifw::core {

class CpuDevice final : public IDevice {
 public:
  Device device() const override { return kCpu; }
  IAllocator& allocator() override { return allocator_; }
  IKernelRegistry& kernels() override { return kernels_; }

  void read_bytes(
      void* dst_host, const void* src_device, size_t bytes
  ) override {
    std::memcpy(dst_host, src_device, bytes);
  }
  void write_bytes(
      void* dst_device, const void* src_host, size_t bytes
  ) override {
    std::memcpy(dst_device, src_host, bytes);
  }

 private:
  CpuAllocator allocator_;
  CpuKernelRegistry kernels_;
};

}  // namespace aifw::core
