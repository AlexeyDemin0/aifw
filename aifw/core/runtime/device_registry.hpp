#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "aifw/core/device/device.hpp"
#include "aifw/core/device/idevice.hpp"
#include "aifw/core/runtime/cpu/cpu_device.hpp"

namespace aifw::core {

struct DeviceHash {
  size_t operator()(const Device& d) const noexcept {
    return std::hash<uint32_t>{}(
        (static_cast<uint32_t>(d.type) << 16) | static_cast<uint32_t>(d.index)
    );
  }
};

class DeviceRegistry {
 public:
  static DeviceRegistry& instance() {
    static DeviceRegistry reg;
    return reg;
  }

  void register_device(std::unique_ptr<IDevice> dev) {
    const Device id = dev->device();
    if (devices_.contains(id))
      throw std::runtime_error(
          "DeviceRegistry: device already registered: " + id.str()
      );
    devices_.emplace(id, std::move(dev));
  }

  IDevice& get(Device id) const {
    auto it = devices_.find(id);
    if (it == devices_.end())
      throw std::runtime_error("DeviceRegistry: device not found: " + id.str());
    return *it->second;
  }

  IDevice& get(const std::string& str) const { return get(parse(str)); }

  bool has(Device id) const { return devices_.contains(id); }

  IDevice& cpu() const { return get(kCpu); }

  DeviceRegistry(const DeviceRegistry&) = delete;
  DeviceRegistry& operator=(const DeviceRegistry&) = delete;

 private:
  DeviceRegistry() { devices_.emplace(kCpu, std::make_unique<CpuDevice>()); }

  static Device parse(const std::string& str) {
    if (str == "cpu") return kCpu;

    auto colon = str.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("DeviceRegistry: invalid device string: " + str);

    const std::string type_str = str.substr(0, colon);
    const int index = std::stoi(str.substr(colon + 1));

    if (type_str == "cuda") return {DeviceType::Cuda, index};

    throw std::runtime_error(
        "DeviceRegistry: unknown device type: " + type_str
    );
  }

  std::unordered_map<Device, std::unique_ptr<IDevice>, DeviceHash> devices_;
};

inline IDevice& cpu() { return DeviceRegistry::instance().cpu(); }
inline IDevice& device(Device id) { return DeviceRegistry::instance().get(id); }
inline IDevice& device(const std::string& str) {
  return DeviceRegistry::instance().get(str);
}

}  // namespace aifw::core
