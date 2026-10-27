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

#include <cmath>
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

bool same_layout(const Tensor& tensor, const Int8TensorInfo& info) {
  if (tensor.dim() != info.rank ||
      static_cast<size_t>(tensor.numel()) != info.bytes)
    return false;
  for (uint8_t d = 0; d < info.rank; ++d) {
    if (tensor.size(d) != info.shape[d])
      return false;
  }
  return executorch::runtime::is_contiguous_dim_order(
      tensor.dim_order().data(), tensor.dim_order().size());
}

bool load_input(
    Int8Interpreter& interpreter,
    size_t index,
    const Tensor& tensor) {
  Int8TensorInfo info{};
  if (!interpreter.input_info(index, info) || !same_layout(tensor, info))
    return false;
  int8_t* destination = interpreter.input_data(index);
  const size_t plane = static_cast<size_t>(info.height) * info.width;
  const size_t channels = static_cast<size_t>(info.channels);
  const bool nchw = info.rank == 4;
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
    const Int8Interpreter& interpreter,
    size_t index,
    Tensor& tensor) {
  Int8TensorInfo info{};
  if (!interpreter.output_info(index, info) || !same_layout(tensor, info))
    return false;
  const int8_t* source = interpreter.output_data(index);
  const size_t plane = static_cast<size_t>(info.height) * info.width;
  const size_t channels = static_cast<size_t>(info.channels);
  const bool nchw = info.rank == 4;
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
  bool profiled = false;
};

class EspNNBackendInterface final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    void* memory = context.get_runtime_allocator()->allocate(
        sizeof(DelegateState), alignof(DelegateState));
    if (memory == nullptr)
      return Error::MemoryAllocationFailed;
    DelegateState* state = new (memory) DelegateState();
    Int8RuntimeOptions options{};
#if CONFIG_ESPNN_DELEGATE_CONSTANTS_TO_PSRAM
    options.copy_constants_to_psram = true;
#endif
#if CONFIG_ESPNN_DELEGATE_DUAL_CORE
    options.dual_core = true;
#endif
    if (!state->interpreter.load(
            static_cast<const uint8_t*>(processed->data()),
            processed->size(),
            options)) {
      state->~DelegateState();
      return Error::DelegateInvalidCompatibility;
    }
    if (state->interpreter.constants_copied())
      processed->Free();
    ++g_stats.delegates;
    g_stats.arena_bytes += state->interpreter.arena_bytes();
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
    for (size_t i = 0; i < inputs; ++i) {
      if (!args[i]->isTensor() ||
          !load_input(interpreter, i, args[i]->toTensor())) {
        std::printf(
            "EspNN delegate: input %u has an unsupported dtype, shape or dim order\n",
            static_cast<unsigned>(i));
        return Error::InvalidArgument;
      }
    }
    if (!interpreter.invoke())
      return Error::Internal;
    for (size_t o = 0; o < outputs; ++o) {
      if (!args[inputs + o]->isTensor())
        return Error::InvalidArgument;
      Tensor tensor = args[inputs + o]->toTensor();
      if (!store_output(interpreter, o, tensor)) {
        std::printf(
            "EspNN delegate: output %u has an unsupported dtype, shape or dim order\n",
            static_cast<unsigned>(o));
        return Error::InvalidArgument;
      }
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

} // namespace espressif
} // namespace backends
} // namespace executorch
