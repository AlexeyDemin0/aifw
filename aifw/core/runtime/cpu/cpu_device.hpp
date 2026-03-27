#pragma once

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

 private:
  CpuAllocator allocator_;
  CpuKernelRegistry kernels_;
};

}  // namespace aifw::core
