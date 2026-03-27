#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace aifw::core {

enum class DeviceType : uint8_t { Cpu, Cuda };

struct Device {
  DeviceType type;
  int index = 0;

  bool operator==(const Device&) const = default;

  std::string str() const {
    switch (type) {
      case DeviceType::Cpu:
        return "cpu";
      case DeviceType::Cuda:
        return "cuda:" + std::to_string(index);
    }
    throw std::runtime_error("Device::str: unknown device type");
  }
};

inline constexpr Device kCpu{DeviceType::Cpu, 0};
inline constexpr Device kCuda{DeviceType::Cuda, 0};

}  // namespace aifw::core
