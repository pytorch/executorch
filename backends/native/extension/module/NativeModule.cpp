// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/native/extension/module/MethodMetaBridge.h>
#include <executorch/backends/native/runtime/MethodMeta.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/Validation.h>
#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/extension/module/ptn_module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::native_module {
namespace {

using internal::PtnModule;
using runtime::Error;

struct ProviderState {
  std::mutex mutex;
  internal::EngineHostFactory factory = nullptr;
  std::weak_ptr<ptn::EngineHost> engine_host;
  std::mutex compile_mutex;
};

ProviderState& provider_state() {
  static ProviderState state;
  return state;
}

std::shared_ptr<ptn::EngineHost> get_engine_host() {
  ProviderState& state = provider_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  std::shared_ptr<ptn::EngineHost> host = state.engine_host.lock();
  if (host == nullptr && state.factory != nullptr) {
    host = state.factory();
    state.engine_host = host;
  }
  return host;
}

// uint64_t-backed transfer buffers must also satisfy double alignment.
static_assert(alignof(uint64_t) >= alignof(double));

template <typename F>
auto catch_boundary(const char* operation, Error exception_error, F&& fn)
    -> decltype(fn()) {
  try {
    return fn();
  } catch (const ptn::ResourceLimitError& error) {
    ET_LOG(Error, "%s exceeded a resource limit: %s", operation, error.what());
    return Error::OutOfResources;
  } catch (const ptn::UnsupportedVersionError& error) {
    ET_LOG(
        Error, "%s found an unsupported version: %s", operation, error.what());
    return Error::NotSupported;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "%s failed: %s", operation, error.what());
    return exception_error;
  } catch (...) {
    ET_LOG(Error, "%s failed with an unknown exception", operation);
    return Error::Internal;
  }
}

executorch::aten::ScalarType to_aten_scalar_type(ptn::ScalarType type) {
  switch (type) {
    case ptn::ScalarType::Byte:
      return executorch::aten::ScalarType::Byte;
    case ptn::ScalarType::Char:
      return executorch::aten::ScalarType::Char;
    case ptn::ScalarType::Short:
      return executorch::aten::ScalarType::Short;
    case ptn::ScalarType::Int:
      return executorch::aten::ScalarType::Int;
    case ptn::ScalarType::Long:
      return executorch::aten::ScalarType::Long;
    case ptn::ScalarType::Half:
      return executorch::aten::ScalarType::Half;
    case ptn::ScalarType::Float:
      return executorch::aten::ScalarType::Float;
    case ptn::ScalarType::Double:
      return executorch::aten::ScalarType::Double;
    case ptn::ScalarType::Bool:
      return executorch::aten::ScalarType::Bool;
    case ptn::ScalarType::BFloat16:
      return executorch::aten::ScalarType::BFloat16;
    case ptn::ScalarType::UInt16:
      return executorch::aten::ScalarType::UInt16;
    case ptn::ScalarType::UInt32:
      return executorch::aten::ScalarType::UInt32;
    case ptn::ScalarType::UInt64:
      return executorch::aten::ScalarType::UInt64;
  }
  throw std::runtime_error("unrecognized PTN scalar type");
}

std::optional<ptn::ScalarType> to_ptn_scalar_type(
    executorch::aten::ScalarType type) {
  switch (type) {
    case executorch::aten::ScalarType::Byte:
      return ptn::ScalarType::Byte;
    case executorch::aten::ScalarType::Char:
      return ptn::ScalarType::Char;
    case executorch::aten::ScalarType::Short:
      return ptn::ScalarType::Short;
    case executorch::aten::ScalarType::Int:
      return ptn::ScalarType::Int;
    case executorch::aten::ScalarType::Long:
      return ptn::ScalarType::Long;
    case executorch::aten::ScalarType::Half:
      return ptn::ScalarType::Half;
    case executorch::aten::ScalarType::Float:
      return ptn::ScalarType::Float;
    case executorch::aten::ScalarType::Double:
      return ptn::ScalarType::Double;
    case executorch::aten::ScalarType::Bool:
      return ptn::ScalarType::Bool;
    case executorch::aten::ScalarType::BFloat16:
      return ptn::ScalarType::BFloat16;
    case executorch::aten::ScalarType::UInt16:
      return ptn::ScalarType::UInt16;
    case executorch::aten::ScalarType::UInt32:
      return ptn::ScalarType::UInt32;
    case executorch::aten::ScalarType::UInt64:
      return ptn::ScalarType::UInt64;
    default:
      return std::nullopt;
  }
}

void validate_et_tensor_info(const ptn::TensorInfo& info) {
  if (info.numel() > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    throw ptn::ResourceLimitError("PTN tensor is too large for TensorImpl");
  }
  if (std::ranges::any_of(info.sizes(), [](const int64_t size) {
        return size > std::numeric_limits<executorch::aten::SizesType>::max();
      })) {
    throw ptn::ResourceLimitError(
        "PTN tensor size is too large for TensorImpl");
  }
  if (std::ranges::any_of(info.strides(), [](const int64_t stride) {
        return stride <
            std::numeric_limits<executorch::aten::StridesType>::min() ||
            stride > std::numeric_limits<executorch::aten::StridesType>::max();
      })) {
    throw ptn::ResourceLimitError(
        "PTN tensor stride is not representable by TensorImpl");
  }
}

size_t storage_words(size_t nbytes) {
  return nbytes / sizeof(uint64_t) + (nbytes % sizeof(uint64_t) != 0);
}

bool has_matching_layout(
    const executorch::aten::Tensor& tensor,
    const ptn::TensorInfo& info) {
#ifndef USE_ATEN_LIB
  const executorch::aten::TensorImpl* const impl = tensor.unsafeGetTensorImpl();
  if (impl == nullptr || !impl->has_layout_metadata()) {
    return false;
  }
#endif
  const auto sizes = tensor.sizes();
  const auto dim_order = tensor.dim_order();
  const auto strides = tensor.strides();
  if (sizes.size() != info.sizes().size() ||
      dim_order.size() != info.dim_order().size() ||
      strides.size() != info.strides().size() ||
      (!sizes.empty() && sizes.data() == nullptr)) {
    return false;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != info.sizes()[i] || dim_order[i] != info.dim_order()[i] ||
        static_cast<int64_t>(strides[i]) != info.strides()[i]) {
      return false;
    }
  }
  return true;
}

Error validate_tensor(
    const runtime::EValue& value,
    const ptn::TensorInfo& info,
    bool require_mutable) {
  if (!value.isTensor()) {
    return Error::InvalidType;
  }
  const executorch::aten::Tensor& tensor = value.toTensor();
  if (!tensor.device().is_cpu()) {
    return Error::NotSupported;
  }
  const std::optional<ptn::ScalarType> dtype =
      to_ptn_scalar_type(tensor.scalar_type());
  if (!dtype || *dtype != info.dtype()) {
    return Error::InvalidArgument;
  }
  if (tensor.dim() < 0 ||
      static_cast<size_t>(tensor.dim()) != info.sizes().size()) {
    return Error::InvalidArgument;
  }
  if (!has_matching_layout(tensor, info) || tensor.nbytes() != info.nbytes()) {
    return Error::InvalidArgument;
  }
  if (info.nbytes() != 0 &&
      (require_mutable ? tensor.mutable_data_ptr() : tensor.const_data_ptr()) ==
          nullptr) {
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

struct InputSlot {
  std::vector<uint64_t> storage;
  bool valid = false;
};

runtime::Result<InputSlot> copy_input(
    const runtime::EValue& value,
    const ptn::TensorInfo& info) {
  const Error validation =
      validate_tensor(value, info, /*require_mutable=*/false);
  if (validation != Error::Ok) {
    return validation;
  }
  InputSlot slot;
  slot.storage.resize(storage_words(info.nbytes()));
  if (info.nbytes() != 0) {
    const void* const data = value.toTensor().const_data_ptr();
    if (data == nullptr) {
      return Error::InvalidArgument;
    }
    std::memcpy(slot.storage.data(), data, info.nbytes());
  }
  slot.valid = true;
  return slot;
}

struct OutputSlot {
  explicit OutputSlot(const ptn::TensorInfo& info)
      : storage(storage_words(info.nbytes())) {
    sizes.reserve(info.sizes().size());
    std::ranges::transform(
        info.sizes(), std::back_inserter(sizes), [](const int64_t size) {
          return static_cast<executorch::aten::SizesType>(size);
        });
    dim_order.reserve(info.dim_order().size());
    std::ranges::transform(
        info.dim_order(), std::back_inserter(dim_order), [](const uint8_t dim) {
          return static_cast<executorch::aten::DimOrderType>(dim);
        });
    strides.reserve(info.strides().size());
    std::ranges::transform(
        info.strides(), std::back_inserter(strides), [](const int64_t stride) {
          return static_cast<executorch::aten::StridesType>(stride);
        });
    impl = std::make_unique<executorch::aten::TensorImpl>(
        to_aten_scalar_type(info.dtype()),
        static_cast<ssize_t>(sizes.size()),
        sizes.data(),
        storage.empty() ? nullptr : storage.data(),
        dim_order.data(),
        strides.data());
    value = runtime::EValue(executorch::aten::Tensor(impl.get()));
  }

  OutputSlot(const OutputSlot&) = delete;
  OutputSlot& operator=(const OutputSlot&) = delete;
  OutputSlot(OutputSlot&&) = delete;
  OutputSlot& operator=(OutputSlot&&) = delete;
  ~OutputSlot() = default;

  std::vector<executorch::aten::SizesType> sizes;
  std::vector<executorch::aten::DimOrderType> dim_order;
  std::vector<executorch::aten::StridesType> strides;
  std::vector<uint64_t> storage;
  std::unique_ptr<executorch::aten::TensorImpl> impl;
  runtime::EValue value;
};

struct OutputBinding {
  void* data = nullptr;
  size_t nbytes = 0;
  bool bound = false;
};

struct MethodState {
  std::unique_ptr<MethodMetaBridge> metadata;
  std::unique_ptr<ptn::EngineExecutable> executable;
  std::vector<ptn::TensorInfo> input_specs;
  std::vector<ptn::TensorInfo> output_specs;
  std::vector<InputSlot> inputs;
  std::vector<std::unique_ptr<OutputSlot>> outputs;
  std::vector<OutputBinding> output_bindings;
  bool outputs_valid = false;
};

class NativePtnModule final : public PtnModule {
 public:
  NativePtnModule(
      std::shared_ptr<const ptn::Package> package,
      std::shared_ptr<const ptn::Program> program,
      std::unordered_map<std::string, MethodState> methods)
      : package_(std::move(package)),
        program_(std::move(program)),
        methods_(std::move(methods)) {}

  runtime::Result<size_t> num_methods() const override {
    return methods_.size();
  }

  runtime::Result<std::unordered_set<std::string>> method_names()
      const override {
    return catch_boundary(
        "listing PTN methods",
        Error::Internal,
        [&]() -> runtime::Result<std::unordered_set<std::string>> {
          std::unordered_set<std::string> names;
          names.reserve(methods_.size());
          for (const auto& entry : methods_) {
            names.insert(entry.first);
          }
          return names;
        });
  }

  runtime::Result<ET_RUNTIME_NAMESPACE::MethodMeta> method_meta(
      const std::string& method_name) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = methods_.find(method_name);
    if (it == methods_.end()) {
      return Error::InvalidArgument;
    }
    try {
      if (it->second.metadata == nullptr) {
        const ptn::Method& method = program_->get_method(method_name);
        it->second.metadata =
            MethodMetaBridge::create(ptn::MethodMeta::from_method(method));
      }
      return it->second.metadata->view();
    } catch (const ptn::ResourceLimitError& error) {
      ET_LOG(Error, "PTN metadata limit: %s", error.what());
      return Error::OutOfResources;
    } catch (const std::bad_alloc&) {
      return Error::MemoryAllocationFailed;
    } catch (const std::exception& error) {
      ET_LOG(Error, "Invalid PTN metadata: %s", error.what());
      return Error::InvalidProgram;
    } catch (...) {
      ET_LOG(Error, "Unknown exception while reading PTN metadata");
      return Error::Internal;
    }
  }

  runtime::Error load_method(const std::string& method_name) override {
    return catch_boundary("loading a PTN method", Error::Internal, [&]() {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = methods_.find(method_name);
      if (it == methods_.end()) {
        return Error::InvalidArgument;
      }
      MethodState& state = it->second;
      if (state.executable != nullptr) {
        return Error::Ok;
      }

      const ptn::Method& method = program_->get_method(method_name);
      const ptn::MethodMeta metadata = ptn::MethodMeta::from_method(method);
      // Preserve ET's InvalidExternalData error before creating engine state.
      const Error constant_validation = catch_boundary(
          "validating PTN constants", Error::InvalidExternalData, [&]() {
            ptn::validate_method_constants(method, *package_);
            return Error::Ok;
          });
      if (constant_validation != Error::Ok) {
        return constant_validation;
      }

      std::unique_ptr<ptn::EngineExecutable> executable;
      const Error compile_error =
          catch_boundary("compiling a PTN method", Error::NotSupported, [&]() {
            std::lock_guard<std::mutex> compile_lock(
                provider_state().compile_mutex);
            if (context_ == nullptr) {
              host_ = get_engine_host();
              if (host_ == nullptr) {
                return Error::NotSupported;
              }
              context_ = host_->create_context(program_, package_);
              if (context_ == nullptr) {
                return Error::Internal;
              }
            }
            executable = context_->compile(method_name);
            return Error::Ok;
          });
      if (compile_error != Error::Ok) {
        return compile_error;
      }
      if (executable == nullptr) {
        return Error::Internal;
      }

      if (executable->num_inputs() != metadata.inputs().size() ||
          executable->num_outputs() != metadata.outputs().size()) {
        return Error::Internal;
      }

      std::vector<ptn::TensorInfo> input_specs;
      std::vector<ptn::TensorInfo> output_specs;
      input_specs.reserve(metadata.inputs().size());
      output_specs.reserve(metadata.outputs().size());
      for (size_t i = 0; i < metadata.inputs().size(); ++i) {
        const ptn::TensorInfo& info = metadata.inputs()[i];
        validate_et_tensor_info(info);
        if (executable->input_dtype(i) != info.dtype() ||
            executable->input_sizes(i) !=
                std::vector<int64_t>(
                    info.sizes().begin(), info.sizes().end())) {
          return Error::Internal;
        }
        input_specs.push_back(info);
      }
      std::vector<std::unique_ptr<OutputSlot>> outputs;
      outputs.reserve(metadata.outputs().size());
      for (size_t i = 0; i < metadata.outputs().size(); ++i) {
        const ptn::TensorInfo& info = metadata.outputs()[i];
        validate_et_tensor_info(info);
        if (executable->output_dtype(i) != info.dtype() ||
            executable->output_sizes(i) !=
                std::vector<int64_t>(
                    info.sizes().begin(), info.sizes().end())) {
          return Error::Internal;
        }
        output_specs.push_back(info);
        outputs.push_back(std::make_unique<OutputSlot>(info));
      }

      state.input_specs = std::move(input_specs);
      state.output_specs = std::move(output_specs);
      state.inputs.resize(state.input_specs.size());
      state.outputs = std::move(outputs);
      state.output_bindings.resize(state.output_specs.size());
      state.outputs_valid = false;
      state.executable = std::move(executable);
      return Error::Ok;
    });
  }

  bool unload_method(const std::string& method_name) override {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = methods_.find(method_name);
      if (it == methods_.end() || it->second.executable == nullptr) {
        return false;
      }
      MethodState& state = it->second;
      state.executable.reset();
      state.input_specs.clear();
      state.output_specs.clear();
      state.inputs.clear();
      state.outputs.clear();
      state.output_bindings.clear();
      state.outputs_valid = false;
      return true;
    } catch (const std::exception& error) {
      ET_LOG(Error, "Unloading a PTN method failed: %s", error.what());
      return false;
    } catch (...) {
      ET_LOG(Error, "Unloading a PTN method failed with an unknown exception");
      return false;
    }
  }

  bool is_method_loaded(const std::string& method_name) const override {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = methods_.find(method_name);
      return it != methods_.end() && it->second.executable != nullptr;
    } catch (...) {
      return false;
    }
  }

  runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name,
      const std::vector<runtime::EValue>& inputs) override {
    return catch_boundary(
        "executing a PTN method",
        Error::Internal,
        [&]() -> runtime::Result<std::vector<runtime::EValue>> {
          std::lock_guard<std::mutex> lock(mutex_);
          MethodState* state = loaded_state(method_name);
          if (state == nullptr) {
            return runtime::Result<std::vector<runtime::EValue>>(
                Error::InvalidState);
          }
          state->outputs_valid = false;
          if (!inputs.empty()) {
            const Error error = set_inputs_locked(*state, inputs);
            if (error != Error::Ok) {
              return runtime::Result<std::vector<runtime::EValue>>(error);
            }
          }
          if (std::ranges::any_of(state->inputs, [](const InputSlot& input) {
                return !input.valid;
              })) {
            return runtime::Result<std::vector<runtime::EValue>>(
                Error::InvalidState);
          }

          for (size_t i = 0; i < state->inputs.size(); ++i) {
            const ptn::TensorInfo& info = state->input_specs[i];
            state->executable->set_input(
                i,
                state->inputs[i].storage.empty()
                    ? nullptr
                    : state->inputs[i].storage.data(),
                info.numel(),
                info.dtype());
          }
          state->executable->execute();

          std::vector<std::vector<uint64_t>> pending;
          pending.reserve(state->output_specs.size());
          for (size_t i = 0; i < state->output_specs.size(); ++i) {
            const ptn::TensorInfo& info = state->output_specs[i];
            pending.emplace_back(storage_words(info.nbytes()));
            state->executable->get_output(
                i,
                pending.back().empty() ? nullptr : pending.back().data(),
                info.numel(),
                info.dtype());
          }

          for (size_t i = 0; i < pending.size(); ++i) {
            const OutputBinding& binding = state->output_bindings[i];
            const size_t nbytes = state->output_specs[i].nbytes();
            if (binding.bound &&
                (binding.nbytes != nbytes ||
                 (nbytes != 0 && binding.data == nullptr))) {
              return runtime::Result<std::vector<runtime::EValue>>(
                  Error::Internal);
            }
          }
          for (size_t i = 0; i < pending.size(); ++i) {
            const size_t nbytes = state->output_specs[i].nbytes();
            if (nbytes != 0) {
              std::memcpy(
                  state->outputs[i]->storage.data(), pending[i].data(), nbytes);
              const OutputBinding& binding = state->output_bindings[i];
              if (binding.bound) {
                std::memcpy(binding.data, pending[i].data(), binding.nbytes);
              }
            }
          }
          state->outputs_valid = true;
          return output_values(*state);
        });
  }

  runtime::Error set_input(
      const std::string& method_name,
      const runtime::EValue& input,
      size_t index) override {
    return catch_boundary("staging a PTN input", Error::Internal, [&]() {
      std::lock_guard<std::mutex> lock(mutex_);
      MethodState* state = loaded_state(method_name);
      if (state == nullptr) {
        return Error::InvalidState;
      }
      if (index >= state->input_specs.size()) {
        return Error::InvalidArgument;
      }
      auto staged = copy_input(input, state->input_specs[index]);
      if (!staged.ok()) {
        return staged.error();
      }
      state->inputs[index] = std::move(*staged);
      return Error::Ok;
    });
  }

  runtime::Error set_inputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& inputs) override {
    return catch_boundary("staging PTN inputs", Error::Internal, [&]() {
      std::lock_guard<std::mutex> lock(mutex_);
      MethodState* state = loaded_state(method_name);
      return state == nullptr ? Error::InvalidState
                              : set_inputs_locked(*state, inputs);
    });
  }

  runtime::Error set_output(
      const std::string& method_name,
      runtime::EValue output,
      size_t index) override {
    return catch_boundary("binding a PTN output", Error::Internal, [&]() {
      std::lock_guard<std::mutex> lock(mutex_);
      MethodState* state = loaded_state(method_name);
      if (state == nullptr) {
        return Error::InvalidState;
      }
      if (index >= state->output_specs.size()) {
        return Error::InvalidArgument;
      }
      const Error validation = validate_tensor(
          output, state->output_specs[index], /*require_mutable=*/true);
      if (validation != Error::Ok) {
        return validation;
      }
      const executorch::aten::Tensor& tensor = output.toTensor();
      state->output_bindings[index] = {
          tensor.mutable_data_ptr(), tensor.nbytes(), /*bound=*/true};
      return Error::Ok;
    });
  }

  runtime::Error set_outputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& outputs) override {
    return catch_boundary("binding PTN outputs", Error::Internal, [&]() {
      std::lock_guard<std::mutex> lock(mutex_);
      MethodState* state = loaded_state(method_name);
      if (state == nullptr) {
        return Error::InvalidState;
      }
      if (outputs.size() != state->output_specs.size()) {
        return Error::InvalidArgument;
      }
      std::vector<OutputBinding> bindings;
      bindings.reserve(outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        const Error validation = validate_tensor(
            outputs[i], state->output_specs[i], /*require_mutable=*/true);
        if (validation != Error::Ok) {
          return validation;
        }
        const executorch::aten::Tensor& tensor = outputs[i].toTensor();
        bindings.push_back(
            {tensor.mutable_data_ptr(), tensor.nbytes(), /*bound=*/true});
      }
      state->output_bindings = std::move(bindings);
      return Error::Ok;
    });
  }

  runtime::Result<std::vector<runtime::EValue>> get_outputs(
      const std::string& method_name) override {
    return catch_boundary(
        "reading PTN outputs",
        Error::Internal,
        [&]() -> runtime::Result<std::vector<runtime::EValue>> {
          std::lock_guard<std::mutex> lock(mutex_);
          const MethodState* state = loaded_state(method_name);
          if (state == nullptr || !state->outputs_valid) {
            return runtime::Result<std::vector<runtime::EValue>>(
                Error::InvalidState);
          }
          return output_values(*state);
        });
  }

  runtime::Result<runtime::EValue> get_output(
      const std::string& method_name,
      size_t index) override {
    return catch_boundary(
        "reading a PTN output",
        Error::Internal,
        [&]() -> runtime::Result<runtime::EValue> {
          std::lock_guard<std::mutex> lock(mutex_);
          MethodState* state = loaded_state(method_name);
          if (state == nullptr || !state->outputs_valid) {
            return runtime::Result<runtime::EValue>(Error::InvalidState);
          }
          if (index >= state->outputs.size()) {
            return runtime::Result<runtime::EValue>(Error::InvalidArgument);
          }
          return runtime::Result<runtime::EValue>(state->outputs[index]->value);
        });
  }

 private:
  MethodState* loaded_state(const std::string& method_name) {
    const auto it = methods_.find(method_name);
    return it == methods_.end() || it->second.executable == nullptr
        ? nullptr
        : &it->second;
  }

  static Error set_inputs_locked(
      MethodState& state,
      const std::vector<runtime::EValue>& inputs) {
    if (inputs.size() != state.input_specs.size()) {
      return Error::InvalidArgument;
    }
    std::vector<InputSlot> staged;
    staged.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto input = copy_input(inputs[i], state.input_specs[i]);
      if (!input.ok()) {
        return input.error();
      }
      staged.push_back(std::move(*input));
    }
    state.inputs = std::move(staged);
    return Error::Ok;
  }

  static std::vector<runtime::EValue> output_values(const MethodState& state) {
    std::vector<runtime::EValue> values;
    values.reserve(state.outputs.size());
    std::ranges::transform(
        state.outputs, std::back_inserter(values), [](const auto& output) {
          return output->value;
        });
    return values;
  }

  std::shared_ptr<const ptn::Package> package_;
  std::shared_ptr<const ptn::Program> program_;
  std::shared_ptr<ptn::EngineHost> host_;
  std::unique_ptr<ptn::EngineContext> context_;
  std::unordered_map<std::string, MethodState> methods_;
  mutable std::mutex mutex_;
};

runtime::Result<std::unique_ptr<PtnModule>> load_ptn(
    runtime::DataLoader& loader,
    ET_RUNTIME_NAMESPACE::Program::Verification verification) {
  try {
    auto size = loader.size();
    if (!size.ok()) {
      return size.error();
    }
    if (*size > ptn::detail::kMaxPackageBytes) {
      return Error::OutOfResources;
    }

    std::vector<uint8_t> bytes(*size);
    if (*size != 0) {
      auto loaded = loader.load(
          0,
          *size,
          runtime::DataLoader::SegmentInfo(
              runtime::DataLoader::SegmentInfo::Type::Program));
      if (!loaded.ok()) {
        return loaded.error();
      }
      if (loaded->size() != *size) {
        return Error::InvalidProgram;
      }
      auto data = loaded->data_safe();
      if (!data.ok()) {
        return data.error();
      }
      if (*data == nullptr) {
        return Error::InvalidProgram;
      }
      std::memcpy(bytes.data(), *data, bytes.size());
    }

    auto package = std::make_shared<ptn::Package>(
        ptn::Package::load(ptn::OwnedBytes::from_vector(std::move(bytes))));
    if (verification ==
        ET_RUNTIME_NAMESPACE::Program::Verification::InternalConsistency) {
      package->verify();
    }
    const ptn::ByteSpan program_bytes = package->program_bytes();
    auto program = std::make_shared<ptn::Program>(
        ptn::Program::load(program_bytes.data(), program_bytes.size()));
    ptn::validate_program_state(*program);

    std::unordered_map<std::string, MethodState> methods;
    const std::vector<std::string> method_names = program->method_names();
    methods.reserve(method_names.size());
    for (const std::string& method_name : method_names) {
      methods.try_emplace(method_name);
    }
    std::unique_ptr<PtnModule> module = std::make_unique<NativePtnModule>(
        std::move(package), std::move(program), std::move(methods));
    return module;
  } catch (const ptn::ResourceLimitError& error) {
    ET_LOG(Error, "PTN package limit: %s", error.what());
    return Error::OutOfResources;
  } catch (const ptn::UnsupportedVersionError& error) {
    ET_LOG(Error, "Unsupported PTN version: %s", error.what());
    return Error::NotSupported;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Invalid PTN package: %s", error.what());
    return Error::InvalidProgram;
  } catch (...) {
    ET_LOG(Error, "Unknown exception while loading PTN package");
    return Error::Internal;
  }
}

const internal::PtnHooks kHooks{load_ptn};

struct RegisterHooks {
  RegisterHooks() {
    const Error error = internal::register_ptn_hooks(kHooks);
    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "PTN Module hook registration failed: 0x%x",
          static_cast<unsigned int>(error));
    }
  }
};

const RegisterHooks kRegisterHooks;

} // namespace

runtime::Error internal::register_engine_host_factory(
    internal::EngineHostFactory factory) {
  if (factory == nullptr) {
    return Error::InvalidArgument;
  }
  ProviderState& state = provider_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.factory == factory) {
    return Error::Ok;
  }
  if (state.factory != nullptr) {
    return Error::AlreadyLoaded;
  }
  state.factory = factory;
  return Error::Ok;
}

} // namespace executorch::extension::native_module

extern "C" void executorch_native_module_ptn_link_anchor() {}
