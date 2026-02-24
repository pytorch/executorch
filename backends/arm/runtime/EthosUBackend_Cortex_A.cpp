/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U Linux driver stack, this relies on the
 * ethos-u-linux-driver-stack for hardware interaction.
 */

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

constexpr int64_t kDefaultEthosUTimeoutNs = 60000000000LL;

struct LinuxDriverOptions {
  std::string device_path = "/dev/ethosu0";
  int64_t timeout_ns = kDefaultEthosUTimeoutNs;
  bool enable_cycle_counter = true;
  std::array<uint32_t, ETHOSU_PMU_EVENT_MAX> pmu_events{};
};

struct PlatformState {
  LinuxDriverOptions options;
};

namespace {

template <typename T>
bool read_scalar_value(const CompileSpec& spec, T* out) {
  if (spec.value.buffer == nullptr || spec.value.nbytes != sizeof(T)) {
    return false;
  }
  std::memcpy(out, spec.value.buffer, sizeof(T));
  return true;
}

std::string read_string_value(const CompileSpec& spec) {
  if (spec.value.buffer == nullptr || spec.value.nbytes == 0) {
    return "";
  }
  const char* raw_begin = static_cast<const char*>(spec.value.buffer);
  const char* raw_end = raw_begin + spec.value.nbytes;
  std::string result(raw_begin, raw_end);
  while (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

LinuxDriverOptions parse_linux_options(ArrayRef<CompileSpec> specs) {
  LinuxDriverOptions options;
  constexpr char kDeviceKey[] = "ethosu.device";
  constexpr char kTimeoutKey[] = "ethosu.timeout_ns";
  constexpr char kCycleCounterKey[] = "ethosu.enable_cycle_counter";
  constexpr char kPmuPrefix[] = "ethosu.pmu_event";

  for (const CompileSpec& spec : specs) {
    if (spec.key == nullptr) {
      continue;
    }

    if (strcmp(spec.key, kDeviceKey) == 0) {
      std::string device_path = read_string_value(spec);
      if (!device_path.empty()) {
        options.device_path = device_path;
      }
      continue;
    }

    if (strcmp(spec.key, kTimeoutKey) == 0) {
      int64_t timeout = 0;
      if (read_scalar_value(spec, &timeout) && timeout > 0) {
        options.timeout_ns = timeout;
      }
      continue;
    }

    if (strcmp(spec.key, kCycleCounterKey) == 0) {
      uint8_t enabled = 0;
      if (read_scalar_value(spec, &enabled)) {
        options.enable_cycle_counter = enabled != 0;
      }
      continue;
    }

    if (strncmp(spec.key, kPmuPrefix, strlen(kPmuPrefix)) == 0) {
      const char* index_str = spec.key + strlen(kPmuPrefix);
      char* endptr = nullptr;
      long idx = std::strtol(index_str, &endptr, 10);
      if (endptr != index_str && idx >= 0 &&
          idx < static_cast<long>(ETHOSU_PMU_EVENT_MAX)) {
        uint32_t event = 0;
        if (read_scalar_value(spec, &event)) {
          options.pmu_events[static_cast<size_t>(idx)] = event;
        }
      }
    }
  }

  return options;
}

class EthosULinuxDeviceCache {
 public:
  EthosU::Device& get(const std::string& device_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!device_ || device_path != active_path_) {
      device_ = std::make_unique<EthosU::Device>(device_path.c_str());
      active_path_ = device_path;
    }
    return *device_;
  }

 private:
  std::mutex mutex_;
  std::string active_path_;
  std::unique_ptr<EthosU::Device> device_;
};

EthosULinuxDeviceCache& get_linux_device_cache() {
  static EthosULinuxDeviceCache cache;
  return cache;
}

const char* inference_status_to_string(EthosU::InferenceStatus status) {
  switch (status) {
    case EthosU::InferenceStatus::OK:
      return "OK";
    case EthosU::InferenceStatus::ERROR:
      return "ERROR";
    case EthosU::InferenceStatus::RUNNING:
      return "RUNNING";
    case EthosU::InferenceStatus::REJECTED:
      return "REJECTED";
    case EthosU::InferenceStatus::ABORTED:
      return "ABORTED";
    case EthosU::InferenceStatus::ABORTING:
      return "ABORTING";
    case EthosU::InferenceStatus::PENDING:
      return "PENDING";
  }
  return "UNKNOWN";
}

Error invoke_linux_driver(
    const VelaHandles& handles,
    const std::vector<const char*>& input_ptrs,
    const std::vector<char*>& output_ptrs,
    const std::vector<size_t>& input_copy_sizes,
    const std::vector<size_t>& output_copy_sizes,
    const LinuxDriverOptions& options) {
  if (handles.outputs == nullptr) {
    ET_LOG(Error, "Ethos-U backend missing output metadata");
    return Error::InvalidProgram;
  }

  try {
    EthosU::Device& device = get_linux_device_cache().get(options.device_path);
    auto network = std::make_shared<EthosU::Network>(
        device,
        reinterpret_cast<const unsigned char*>(handles.cmd_data),
        handles.cmd_data_size);

    std::shared_ptr<EthosU::Buffer> constant_buffer =
        std::make_shared<EthosU::Buffer>();
    if (handles.weight_data_size > 0) {
      auto constant_buffers = device.createBuffers({handles.weight_data_size});
      constant_buffer = constant_buffers.front();
      constant_buffer->write(
          const_cast<char*>(handles.weight_data), handles.weight_data_size);
    }

    std::shared_ptr<EthosU::Buffer> intermediate_buffer =
        std::make_shared<EthosU::Buffer>();
    if (handles.scratch_data_size > 0) {
      auto scratch_buffers = device.createBuffers({handles.scratch_data_size});
      intermediate_buffer = scratch_buffers.front();
    }

    std::vector<std::shared_ptr<EthosU::Buffer>> ifm_buffers;
    if (handles.inputs != nullptr && handles.inputs->count > 0) {
      if (input_copy_sizes.size() !=
          static_cast<size_t>(handles.inputs->count)) {
        ET_LOG(
            Error,
            "Mismatch between input metadata (%d) and copy plan (%zu)",
            handles.inputs->count,
            input_copy_sizes.size());
        return Error::InvalidProgram;
      }
      if (input_ptrs.size() != input_copy_sizes.size()) {
        ET_LOG(
            Error,
            "Mismatch between input metadata and runtime pointers (%zu vs %zu)",
            input_ptrs.size(),
            input_copy_sizes.size());
        return Error::InvalidState;
      }
      ifm_buffers = device.createBuffers(input_copy_sizes);
      for (int i = 0; i < handles.inputs->count; ++i) {
        const size_t copy_size = input_copy_sizes[i];
        if (copy_size == 0) {
          continue;
        }
        const char* src = input_ptrs[i];
        if (src == nullptr) {
          ET_LOG(Error, "Missing input buffer for index %d", i);
          return Error::InvalidState;
        }
        ifm_buffers[i]->write(const_cast<char*>(src), copy_size);
      }
    }

    if (output_copy_sizes.size() !=
        static_cast<size_t>(handles.outputs->count)) {
      ET_LOG(
          Error,
          "Mismatch between output metadata (%d) and copy plan (%zu)",
          handles.outputs->count,
          output_copy_sizes.size());
      return Error::InvalidProgram;
    }
    if (output_ptrs.size() != output_copy_sizes.size()) {
      ET_LOG(
          Error,
          "Mismatch between output metadata and runtime buffers (%zu vs %zu)",
          output_ptrs.size(),
          output_copy_sizes.size());
      return Error::InvalidState;
    }
    auto ofm_buffers = device.createBuffers(output_copy_sizes);

    auto inference = std::make_unique<EthosU::Inference>(
        network,
        ifm_buffers.begin(),
        ifm_buffers.end(),
        ofm_buffers.begin(),
        ofm_buffers.end(),
        intermediate_buffer,
        constant_buffer,
        options.pmu_events,
        options.enable_cycle_counter);

    if (inference->wait(options.timeout_ns)) {
      ET_LOG(
          Error,
          "Ethos-U inference timed out after %lld ns",
          static_cast<long long>(options.timeout_ns));
      return Error::InvalidState;
    }

    auto status = inference->status();
    if (status != EthosU::InferenceStatus::OK) {
      ET_LOG(
          Error,
          "Ethos-U inference failed with status %s",
          inference_status_to_string(status));
      return Error::InvalidState;
    }

    if (options.enable_cycle_counter) {
      try {
        uint64_t cycles = inference->getCycleCounter();
        ET_LOG(
            Info,
            "Ethos-U Linux delegate cycle counter: %llu",
            static_cast<unsigned long long>(cycles));
      } catch (const std::exception& e) {
        ET_LOG(Debug, "Failed to read Ethos-U cycle counter: %s", e.what());
      }
    }

    for (int i = 0; i < handles.outputs->count; ++i) {
      const size_t copy_size = output_copy_sizes[i];
      if (copy_size == 0) {
        continue;
      }
      char* dst = output_ptrs[i];
      if (dst == nullptr) {
        ET_LOG(Error, "Missing output buffer for index %d", i);
        return Error::InvalidState;
      }
      ofm_buffers[i]->read(dst, copy_size);
    }
  } catch (const std::exception& e) {
    ET_LOG(Error, "Ethos-U Linux driver invocation failed: %s", e.what());
    return Error::InvalidState;
  }

  return Error::Ok;
}
} // namespace

PlatformState* platform_init(
    ArrayRef<CompileSpec> specs,
    MemoryAllocator* allocator) {
  (void)allocator;
  PlatformState* state = new (std::nothrow) PlatformState();
  if (state == nullptr) {
    return nullptr;
  }
  state->options = parse_linux_options(specs);
  return state;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* execution_handle,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* /*ethosu_scratch*/) {
  std::vector<size_t> input_copy_sizes;
  std::vector<const char*> linux_input_ptrs;
  if (input_count > 0) {
    input_copy_sizes.resize(input_count, 0);
    linux_input_ptrs.resize(input_count, nullptr);
  }

  std::vector<size_t> output_io_bytes;
  std::vector<char*> linux_output_ptrs;
  if (output_count > 0) {
    output_io_bytes.resize(output_count, 0);
    linux_output_ptrs.resize(output_count, nullptr);
  }

  for (int i = 0; i < input_count; ++i) {
    auto tensor_in = args[i]->toTensor();
    linux_input_ptrs[i] = tensor_in.const_data_ptr<char>();
    input_copy_sizes[i] = tensor_in.nbytes();
  }

  if (handles.outputs != nullptr) {
    for (int i = 0; i < output_count; ++i) {
      int tensor_count = 1, io_count = 1;
      auto tensor_out = args[input_count + i]->toTensor();
      calculate_dimensions(
          tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);
      if (i < static_cast<int>(output_io_bytes.size())) {
        output_io_bytes[i] = static_cast<size_t>(io_count) *
            static_cast<size_t>(handles.outputs->io[i].elem_size);
      }
      const size_t tensor_nbytes = tensor_out.nbytes();
      if (i < static_cast<int>(output_io_bytes.size()) &&
          output_io_bytes[i] != tensor_nbytes) {
        ET_LOG(
            Error,
            "Ethos-U Linux backend output size mismatch for index %d: "
            "driver IO bytes = %zu, tensor bytes = %zu",
            i,
            output_io_bytes[i],
            tensor_nbytes);
        return Error::InvalidState;
      }
      linux_output_ptrs[i] = tensor_out.mutable_data_ptr<char>();
    }
  }

  const PlatformState* state = execution_handle->platform_state;
  if (state == nullptr) {
    ET_LOG(Error, "Ethos-U Linux backend missing platform state");
    return Error::InvalidState;
  }

  return invoke_linux_driver(
      handles,
      linux_input_ptrs,
      linux_output_ptrs,
      input_copy_sizes,
      output_io_bytes,
      state->options);
}

} // namespace arm
} // namespace backends
} // namespace executorch
