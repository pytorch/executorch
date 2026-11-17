/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <espnn/espnn_backend.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include <espnn/int8_interpreter.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace espressif {
namespace {

DelegateStats g_stats{};
// Incremented by each reset request; a delegate whose state is older restores
// its initial state before its next execute.
std::atomic<uint32_t> g_state_generation{0};

constexpr float kHalfEvenRoundingBias = 12582912.0f;

float round_half_even_below_2p22(float value) {
  return (value + kHalfEvenRoundingBias) - kHalfEvenRoundingBias;
}

int8_t quantize(float value, float inverse_scale, int32_t zero_point) {
  const float scaled = value * inverse_scale;
  if (!std::isfinite(scaled))
    return scaled > 0.0f ? 127 : -128;
  if (scaled > 1.0e6f)
    return 127;
  if (scaled < -1.0e6f)
    return -128;
  int32_t q =
      static_cast<int32_t>(round_half_even_below_2p22(scaled)) + zero_point;
  if (q < -128)
    q = -128;
  if (q > 127)
    q = 127;
  return static_cast<int8_t>(q);
}

// How an ExecuTorch tensor's bytes relate to the delegate's device layout
// (NHWC for rank 4). kDevice: identical, because the tensor is not rank 4, its
// transpose is trivial (one channel, or one pixel), or it is channels-last.
// kNchw: a contiguous rank-4 tensor that needs transposing.
enum class Layout : uint8_t { kInvalid, kNchw, kDevice };

Layout tensor_layout(const Tensor& tensor, const Int8TensorInfo& info) {
  if (tensor.dim() != info.rank ||
      static_cast<size_t>(tensor.numel()) != info.bytes)
    return Layout::kInvalid;
  for (uint8_t d = 0; d < info.rank; ++d) {
    if (tensor.size(d) != info.shape[d])
      return Layout::kInvalid;
  }
  const auto dim_order = tensor.dim_order();
  if (executorch::runtime::is_contiguous_dim_order(
          dim_order.data(), dim_order.size())) {
    const bool trivial = info.rank != 4 || info.channels == 1 ||
        static_cast<int64_t>(info.height) * info.width == 1;
    return trivial ? Layout::kDevice : Layout::kNchw;
  }
  if (info.rank == 4 &&
      executorch::runtime::is_channels_last_dim_order(
          dim_order.data(), dim_order.size()))
    return Layout::kDevice;
  return Layout::kInvalid;
}

Error missing_data(const char* kind, size_t index) {
  std::printf(
      "EspNN delegate: %s %u has no data\n",
      kind,
      static_cast<unsigned>(index));
  return Error::InvalidArgument;
}

// The kernels may rely on the activation arena's 16-byte alignment, so only
// buffers aligned like it are used in place.
bool arena_aligned(const void* data) {
  return reinterpret_cast<uintptr_t>(data) % 16 == 0;
}

bool load_input(
    const Int8TensorInfo& info,
    Layout layout,
    const Tensor& tensor,
    int8_t* destination) {
  const size_t plane = static_cast<size_t>(info.height) * info.width;
  const size_t channels = static_cast<size_t>(info.channels);
  const bool nchw = layout == Layout::kNchw;
  if (tensor.scalar_type() == ScalarType::Float) {
    const float* source = tensor.const_data_ptr<float>();
    const float inverse =
        static_cast<float>(1.0 / static_cast<double>(info.scale));
    if (!nchw) {
      for (size_t i = 0; i < info.bytes; ++i)
        destination[i] = quantize(source[i], inverse, info.zero_point);
      return true;
    }
    for (size_t p = 0; p < plane; ++p) {
      for (size_t c = 0; c < channels; ++c) {
        destination[p * channels + c] =
            quantize(source[c * plane + p], inverse, info.zero_point);
      }
    }
    return true;
  }
  if (tensor.scalar_type() == ScalarType::Char) {
    const int8_t* source = tensor.const_data_ptr<int8_t>();
    if (!nchw) {
      std::memcpy(destination, source, info.bytes);
      return true;
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t p = 0; p < plane; ++p)
        destination[p * channels + c] = source[c * plane + p];
    }
    return true;
  }
  return false;
}

bool store_output(
    const Int8TensorInfo& info,
    Layout layout,
    const int8_t* source,
    Tensor& tensor) {
  const size_t plane = static_cast<size_t>(info.height) * info.width;
  const size_t channels = static_cast<size_t>(info.channels);
  const bool nchw = layout == Layout::kNchw;
  if (tensor.scalar_type() == ScalarType::Float) {
    float* destination = tensor.mutable_data_ptr<float>();
    if (!nchw) {
      for (size_t i = 0; i < info.bytes; ++i) {
        destination[i] =
            static_cast<float>(source[i] - info.zero_point) * info.scale;
      }
      return true;
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t p = 0; p < plane; ++p) {
        destination[c * plane + p] =
            static_cast<float>(source[p * channels + c] - info.zero_point) *
            info.scale;
      }
    }
    return true;
  }
  if (tensor.scalar_type() == ScalarType::Char) {
    int8_t* destination = tensor.mutable_data_ptr<int8_t>();
    if (!nchw) {
      if (destination != source)
        std::memcpy(destination, source, info.bytes);
      return true;
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t p = 0; p < plane; ++p)
        destination[c * plane + p] = source[p * channels + c];
    }
    return true;
  }
  return false;
}

struct DelegateState {
  Int8Interpreter interpreter;
  uint32_t conversion_slots = 0;
  bool profiled = false;
  uint32_t state_generation = 0;
  // Copies of outputs outside the arena whose memory is misaligned, in output
  // order.
  int8_t* staging = nullptr;

  ~DelegateState() {
    if (staging != nullptr)
      heap_caps_free(staging);
  }

  // A graph tensor returned more than once is written where its first
  // output is bound, and a graph input returned as an output stays where
  // the input is bound; both are copied to their outputs afterwards.
  bool returned_again(size_t o) const {
    Int8TensorInfo info{};
    Int8TensorInfo other{};
    if (!interpreter.output_info(o, info))
      return false;
    for (size_t earlier = 0; earlier < o; ++earlier) {
      if (interpreter.output_info(earlier, other) &&
          other.tensor_id == info.tensor_id)
        return true;
    }
    for (size_t i = 0; i < interpreter.input_count(); ++i) {
      if (interpreter.input_info(i, other) && other.tensor_id == info.tensor_id)
        return true;
    }
    return false;
  }

  // Offset of output end's staging copy, the staging buffer's size for
  // end = output_count(), or SIZE_MAX if that overflows.
  size_t staging_offset(size_t end) const {
    const size_t inputs = interpreter.input_count();
    size_t total = 0;
    for (size_t o = 0; o < end; ++o) {
      Int8TensorInfo info{};
      if ((conversion_slots & (1U << (inputs + o))) != 0 || returned_again(o) ||
          !interpreter.output_info(o, info))
        continue;
      if (info.bytes > SIZE_MAX - 15 - total)
        return SIZE_MAX;
      total += (info.bytes + 15) & ~static_cast<size_t>(15);
    }
    return total;
  }

  // Only misaligned outputs use the staging buffer, so it goes to PSRAM when
  // there is any and leaves internal RAM to the arena.
  bool allocate_staging() {
    const size_t total = staging_offset(interpreter.output_count());
    if (total == 0)
      return true;
    void* memory = nullptr;
    if (total != SIZE_MAX) {
      memory = heap_caps_aligned_alloc(
          16, total, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      if (memory == nullptr) {
        memory = heap_caps_aligned_alloc(
            16, total, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      }
    }
    if (memory == nullptr) {
      std::printf(
          "EspNN delegate: cannot allocate %u staging bytes\n",
          static_cast<unsigned>(total));
      return false;
    }
    staging = static_cast<int8_t*>(memory);
    std::memset(staging, 0, total);
    g_stats.staging_bytes += total;
    return true;
  }
};

class EspNNBackendInterface final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    Int8RuntimeOptions options{};
    for (const auto& spec : compile_specs) {
      if (std::strcmp(spec.key, "espnn_io_conversion_slots") == 0) {
        if (spec.value.buffer == nullptr || spec.value.nbytes != 4)
          return Error::DelegateInvalidCompatibility;
        options.external_io = true;
        options.conversion_slots =
            read_le_u32(static_cast<const uint8_t*>(spec.value.buffer));
      }
    }
    void* memory = context.get_runtime_allocator()->allocate(
        sizeof(DelegateState), alignof(DelegateState));
    if (memory == nullptr)
      return Error::MemoryAllocationFailed;
    DelegateState* state = new (memory) DelegateState();
#if CONFIG_ESPNN_DELEGATE_CONSTANTS_TO_PSRAM
    options.copy_constants_to_psram = true;
#endif
#if CONFIG_ESPNN_DELEGATE_DUAL_CORE
    options.dual_core = true;
#endif
#if CONFIG_ESPNN_DELEGATE_PROFILE
    options.profile_operations = true;
#endif
#if CONFIG_ESPNN_DELEGATE_SCRATCH_IN_ARENA
    options.scratch_in_arena = true;
#endif
    if (!state->interpreter.load(
            static_cast<const uint8_t*>(processed->data()),
            processed->size(),
            options)) {
      state->~DelegateState();
      return Error::DelegateInvalidCompatibility;
    }
    // Older exports use the arena for every I/O slot.
    state->conversion_slots = options.external_io
        ? state->interpreter.conversion_slots()
        : UINT32_MAX;
    if (!state->allocate_staging()) {
      state->~DelegateState();
      return Error::MemoryAllocationFailed;
    }
    if (state->interpreter.constants_copied())
      processed->Free();
    state->state_generation =
        g_state_generation.load(std::memory_order_relaxed);
    ++g_stats.delegates;
    g_stats.arena_bytes += state->interpreter.arena_bytes();
    g_stats.scratch_bytes += state->interpreter.scratch_bytes();
    g_stats.parameter_bytes += state->interpreter.parameter_bytes();
    g_stats.table_bytes += state->interpreter.table_bytes();
    g_stats.states += state->interpreter.state_count();
    g_stats.states_updated_in_place +=
        state->interpreter.states_updated_in_place();
    return state;
  }

  Error execute(
      BackendExecutionContext&,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    DelegateState* state = static_cast<DelegateState*>(handle);
    Int8Interpreter& interpreter = state->interpreter;
    const size_t inputs = interpreter.input_count();
    const size_t outputs = interpreter.output_count();
    if (args.size() != inputs + outputs) {
      std::printf(
          "EspNN delegate: expected %u arguments, got %u\n",
          static_cast<unsigned>(inputs + outputs),
          static_cast<unsigned>(args.size()));
      return Error::InvalidArgument;
    }
    const int64_t started = esp_timer_get_time();
    if (interpreter.state_count() != 0) {
      const uint32_t generation =
          g_state_generation.load(std::memory_order_relaxed);
      if (generation != state->state_generation) {
        interpreter.reset_states();
        state->state_generation = generation;
      }
    }
    // Inputs are copied into their arena slots, so outputs may be written over
    // the caller's input buffers. Output conversion buffers share the arena;
    // compatible outputs are bound directly, independently of ExecuTorch's
    // output memory planning.
    Int8TensorInfo info{};
    for (size_t i = 0; i < inputs; ++i) {
      const Layout layout =
          args[i]->isTensor() && interpreter.input_info(i, info)
          ? tensor_layout(args[i]->toTensor(), info)
          : Layout::kInvalid;
      if (layout != Layout::kInvalid &&
          args[i]->toTensor().const_data_ptr() == nullptr)
        return missing_data("input", i);
      const bool ok =
          layout != Layout::kInvalid &&
          load_input(
              info, layout, args[i]->toTensor(), interpreter.input_data(i));
      if (!ok) {
        std::printf(
            "EspNN delegate: input %u has an unsupported dtype, shape or dim order\n",
            static_cast<unsigned>(i));
        return Error::InvalidArgument;
      }
    }
    uint32_t staged_outputs = 0;
    for (size_t o = 0; o < outputs; ++o) {
      const Layout layout =
          args[inputs + o]->isTensor() && interpreter.output_info(o, info)
          ? tensor_layout(args[inputs + o]->toTensor(), info)
          : Layout::kInvalid;
      bool ok = layout != Layout::kInvalid;
      if (ok && args[inputs + o]->toTensor().const_data_ptr() == nullptr)
        return missing_data("output", o);
      if (ok && (state->conversion_slots & (1U << (inputs + o))) != 0) {
        staged_outputs |= 1U << o;
        continue;
      }
      if (ok && state->returned_again(o)) {
        staged_outputs |= 1U << o;
      } else if (ok) {
        Tensor tensor = args[inputs + o]->toTensor();
        if (tensor.scalar_type() == ScalarType::Char &&
            layout == Layout::kDevice &&
            arena_aligned(tensor.const_data_ptr())) {
          ok = interpreter.bind_output(o, tensor.mutable_data_ptr<int8_t>());
        } else {
          ok = interpreter.bind_output(
              o, state->staging + state->staging_offset(o));
          staged_outputs |= 1U << o;
          ++g_stats.staged_outputs;
        }
      }
      if (!ok) {
        std::printf(
            "EspNN delegate: output %u has an unsupported dtype, shape or dim order\n",
            static_cast<unsigned>(o));
        return Error::InvalidArgument;
      }
    }
    if (!interpreter.invoke())
      return Error::Internal;
    for (size_t o = 0; o < outputs; ++o) {
      if ((staged_outputs & (1U << o)) == 0)
        continue;
      Tensor tensor = args[inputs + o]->toTensor();
      interpreter.output_info(o, info);
      if (!store_output(
              info,
              tensor_layout(tensor, info),
              interpreter.output_data(o),
              tensor))
        return Error::InvalidArgument;
    }
    ++g_stats.calls;
    g_stats.total_us += esp_timer_get_time() - started;
#if CONFIG_ESPNN_DELEGATE_PROFILE
    if (!state->profiled) {
      state->profiled = true;
      std::printf(
          "EspNN delegate profile (%u ops, %lld us):\n",
          static_cast<unsigned>(interpreter.graph().header.num_operations),
          static_cast<long long>(interpreter.last_invoke_us()));
      interpreter.print_profile();
    }
#endif
    return Error::Ok;
  }

  Error set_option(
      BackendOptionContext&,
      const Span<BackendOption>& backend_options) override {
    for (const BackendOption& option : backend_options) {
      const bool* value = std::get_if<bool>(&option.value);
      if (std::strcmp(option.key, kResetStatesOption) != 0 ||
          value == nullptr) {
        std::printf("EspNN delegate: unknown option %s\n", option.key);
        return Error::NotSupported;
      }
      if (*value)
        reset_delegate_states();
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr)
      static_cast<DelegateState*>(handle)->~DelegateState();
  }
};

EspNNBackendInterface g_backend;
bool g_registered = executorch::runtime::register_backend(
                        Backend{kEspNNBackendId, &g_backend}) == Error::Ok;

} // namespace

[[maybe_unused]] bool register_espnn_backend() {
  if (!g_registered) {
    g_registered = executorch::runtime::register_backend(
                       Backend{kEspNNBackendId, &g_backend}) == Error::Ok;
  }
  return g_registered;
}

DelegateStats delegate_stats() {
  return g_stats;
}

[[maybe_unused]] void reset_delegate_stats() {
  g_stats = {};
}

void reset_delegate_states() {
  g_state_generation.fetch_add(1, std::memory_order_relaxed);
}

} // namespace espressif
} // namespace backends
} // namespace executorch
