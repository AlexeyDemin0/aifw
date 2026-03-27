#include <memory>
#include <stdexcept>
#include <string>

#include "../framework/test.hpp"
#include "aifw/core/device/device.hpp"
#include "aifw/core/device/idevice.hpp"
#include "aifw/core/runtime/cpu/cpu_device.hpp"
#include "aifw/core/runtime/device_registry.hpp"

using namespace aifw::core;

TEST(Device, cpu_str) { EXPECT_EQ(kCpu.str(), std::string("cpu")); }

TEST(Device, cuda_str) {
  Device d{DeviceType::Cuda, 1};
  EXPECT_EQ(d.str(), std::string("cuda:1"));
}

TEST(Device, equality) {
  EXPECT_TRUE(kCpu == kCpu);
  EXPECT_TRUE((Device{DeviceType::Cuda, 0}) == (Device{DeviceType::Cuda, 0}));
  EXPECT_TRUE((Device{DeviceType::Cuda, 0}) != (Device{DeviceType::Cuda, 1}));
  EXPECT_TRUE(kCpu != (Device{DeviceType::Cuda, 0}));
}

TEST(DeviceRegistry, cpu_always_available) {
  IDevice& d = cpu();
  EXPECT_TRUE(d.device() == kCpu);
}

TEST(DeviceRegistry, get_by_string) {
  IDevice& d = device("cpu");
  EXPECT_TRUE(d.device() == kCpu);
}

TEST(DeviceRegistry, get_unknown_throws) {
  EXPECT_THROWS(std::runtime_error, []() { device("cuda:0"); });
}

TEST(DeviceRegistry, get_invalid_string_throws) {
  EXPECT_THROWS(std::runtime_error, []() { device("notadevice"); });
}

TEST(DeviceRegistry, double_register_throws) {
  EXPECT_THROWS(std::runtime_error, []() {
    DeviceRegistry::instance().register_device(std::make_unique<CpuDevice>());
  });
}
