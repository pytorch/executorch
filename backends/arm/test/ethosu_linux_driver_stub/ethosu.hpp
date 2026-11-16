/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// Match the Linux driver instance API even when the stub needs no state.
// cppcheck-suppress-begin functionStatic
namespace EthosU {
struct HardwareConfiguration {
  enum class DeviceType { SUBSYSTEM, DIRECT };
  DeviceType type = DeviceType::DIRECT;
  uint32_t macsPerClockCycle = 8;
};
struct Capabilities {
  struct {
    struct {
      uint32_t major = 6;
    } product;
  } hwId;
  HardwareConfiguration hwCfg;
};
class Buffer {
 public:
  int getFd() const {
    return -1;
  }
  size_t size() const {
    return 0;
  }
  void write(void*, size_t) const {}
  void read(void*, size_t) const {}
};
class Device {
 public:
  inline static Capabilities caps{};
  inline static bool fail_capabilities = false;
  explicit Device(const char*) {}
  Capabilities capabilities() const {
    if (fail_capabilities) {
      throw std::runtime_error("Capabilities failed");
    }
    return caps;
  }
  std::vector<std::shared_ptr<Buffer>> createBuffers(
      const std::vector<size_t>&) const {
    return {std::make_shared<Buffer>()};
  }
};
class Network {
 public:
  Network(Device&, const unsigned char*, size_t) {}
};
enum class InferenceStatus {
  OK,
  ERROR,
  RUNNING,
  REJECTED,
  ABORTED,
  ABORTING,
  PENDING
};

class Inference {
 public:
  inline static bool throw_on_submit = false;
  inline static bool throw_on_wait = false;
  inline static bool timeout = false;
  inline static InferenceStatus result = InferenceStatus::OK;
  inline static bool cycle_counter_enabled = false;

  Inference() = default;
  template <typename Iterator>
  Inference(
      std::shared_ptr<Network>,
      Iterator,
      Iterator,
      Iterator,
      Iterator,
      std::shared_ptr<Buffer>,
      std::shared_ptr<Buffer>,
      const std::array<uint32_t, 8>&,
      bool enable_cycles) {
    if (throw_on_submit) {
      throw std::runtime_error("Submission failed");
    }
    cycle_counter_enabled = enable_cycles;
  }
  static uint32_t getMaxPmuEventCounters() {
    return 8;
  }
  bool wait(int64_t) const {
    if (throw_on_wait) {
      throw std::runtime_error("Wait failed");
    }
    return timeout;
  }
  InferenceStatus status() const {
    return result;
  }
  uint64_t cycles = 0;
  std::array<uint64_t, 8> counters{};
  bool fail_read = false;

  uint64_t getCycleCounter() const {
    if (fail_read) {
      throw std::runtime_error("PMU read failed");
    }
    return cycles;
  }
  const std::array<uint64_t, 8>& getPmuCounters() const {
    return counters;
  }
};
} // namespace EthosU
// cppcheck-suppress-end functionStatic
