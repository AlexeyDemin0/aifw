#pragma once

#include "../ops/cpu_kernel_registry.hpp"
#include "cpu_allocator.hpp"
#include "iallocator.hpp"
#include "idevice.hpp"
#include "ikernel_registry.hpp"

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
