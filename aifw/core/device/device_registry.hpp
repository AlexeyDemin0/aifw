#pragma once

#include <memory>
#include <string>

#include "aifw/core/device/device.hpp"
#include "aifw/core/device/idevice.hpp"

namespace aifw::core {

class DeviceRegistry {
 public:
  static DeviceRegistry& instance();

  void register_device(std::unique_ptr<aifw::core::IDevice> dev);
  IDevice& get(Device id) const;
  IDevice& get(const std::string& str) const;
  bool has(Device id) const;

  IDevice& cpu() const { return get(kCpu); }

 private:
  DeviceRegistry();
};

}  // namespace aifw::core
