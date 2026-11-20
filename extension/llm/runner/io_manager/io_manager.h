/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * @brief Base class for managing input/output operations for LLM inference.
 *
 * IOManager provides an interface for handling the input preparation and
 * output processing for both prefill and decode phases of LLM inference.
 * Derived classes must implement the virtual methods to provide specific IO
 * management functionality.
 */
class ET_EXPERIMENTAL IOManager {
 public:
  /**
   * @brief Construct an IOManager bound to a Module.
   *
   * @param module The Module used for querying method metadata and execution.
   */
  explicit IOManager(ET_MODULE_NAMESPACE::Module& module) : module_(module) {}

  /**
   * @brief Virtual destructor to allow proper cleanup in derived classes.
   */
  virtual ~IOManager() = default;

  /**
   * @brief Load the IO manager with method metadata for prefill and
   * decode operations.
   *
   * @param prefill_method The prefill method to initialize with.
   * @param decode_method The decode method to initialize with.
   */
  ET_NODISCARD virtual runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) {
    (void)prefill_method;
    (void)decode_method;
    return runtime::Error::Ok;
  }

  /**
   * @brief Load the IO manager using the default method names.
   *
   * Uses "forward" for both prefill and decode.
   *
   * @return Error code.
   */
  ET_NODISCARD runtime::Error load() {
    return load("forward", "forward");
  }

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) {
    (void)prefill_method;
    (void)decode_method;
    return runtime::Error::Ok;
  }

  /**
   * @brief Reset the IO manager state using the default method names.
   *
   * Uses "forward" for both prefill and decode.
   *
   * @return Error code.
   */
  ET_NODISCARD runtime::Error reset() {
    return reset("forward", "forward");
  }

  /**
   * @brief Prepare inputs for the prefill phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param prefill_method The prefill method to prepare inputs for.
   * @return std::vector<runtime::EValue> Vector of prepared inputs
   * for the prefill method.
   */
  virtual runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& prefill_method) {
    auto method_meta = module_.method_meta(prefill_method);
    if (!method_meta.ok()) {
      return method_meta.error();
    }
    if (method_meta->num_inputs() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for prefill method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          method_meta->num_inputs());
      return runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<runtime::EValue>{input, start_pos};
  }

  /**
   * @brief Prepare inputs for the prefill phase using the default method name.
   *
   * Uses "forward" as the prefill method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the prefill method.
   */
  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos) {
    return prepare_prefill(input, start_pos, "forward");
  }

  /**
   * @brief Prepare inputs for the decode phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param decode_method The decode method to prepare inputs for.
   * @return std::vector<runtime::EValue> Vector of prepared inputs
   * for the decode method.
   */
  virtual runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& decode_method) {
    auto method_meta = module_.method_meta(decode_method);
    if (!method_meta.ok()) {
      return method_meta.error();
    }
    if (method_meta->num_inputs() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for decode method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          method_meta->num_inputs());
      return runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<runtime::EValue>{input, start_pos};
  }

  /**
   * @brief Prepare inputs for the decode phase using the default method name.
   *
   * Uses "forward" as the decode method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the decode method.
   */
  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos) {
    return prepare_decode(input, start_pos, "forward");
  }

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& prefill_method) {
    (void)model_outputs;
    (void)prefill_method;
    // No post inference work to do.
    return runtime::Error::Ok;
  }

  /**
   * @brief Process outputs from the prefill phase using the default method.
   *
   * Uses "forward" as the prefill method.
   *
   * @param model_outputs Vector of outputs from the prefill execution.
   * @return Error code.
   */
  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_prefill(model_outputs, "forward");
  }

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) {
    (void)model_outputs;
    (void)decode_method;
    // No post inference work to do.
    return runtime::Error::Ok;
  }

  /**
   * @brief Process outputs from the decode phase using the default method.
   *
   * Uses "forward" as the decode method.
   *
   * @param model_outputs Vector of outputs from the decode execution.
   * @return Error code.
   */
  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs) {
    return update_decode(model_outputs, "forward");
  }

  /**
   * @brief The logits among a method's outputs, as the sampler reads them.
   *
   * The default takes a method that returns only its logits. An IOManager for
   * a model that returns more (e.g. its updated cache) picks them out, and may
   * hand back a converted copy (e.g. dequantized).
   *
   * @param model_outputs The outputs of the method just executed.
   * @param method The method that produced them.
   * @return The logits tensor.
   */
  virtual runtime::Result<executorch::aten::Tensor> logits(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& method) {
    (void)method;
    if (model_outputs.size() != 1 || !model_outputs[0].isTensor()) {
      ET_LOG(
          Error,
          "Expected the method to return only its logits tensor, got %zu outputs.",
          model_outputs.size());
      return runtime::Error::InvalidState;
    }
    return model_outputs[0].toTensor();
  }

  /**
   * @brief The most tokens one call of `method` takes, or -1 for no limit of
   * the IOManager's own (e.g. a fixed-size prompt step).
   */
  virtual int64_t max_tokens_per_step(const std::string& method) const {
    (void)method;
    return -1;
  }

  virtual int64_t max_tokens_per_step(
      const std::string& method,
      int64_t start_pos) const {
    (void)start_pos;
    return max_tokens_per_step(method);
  }

  /**
   * @brief For an encoder-decoder contract: route `encoder_method`'s outputs
   * into the decode step's fixed inputs. Call once after loading, then after
   * each run of the encoder pass its outputs to `update_encoder`.
   */
  ET_NODISCARD virtual runtime::Error load_encoder(
      const std::string& encoder_method) {
    (void)encoder_method;
    return runtime::Error::NotSupported;
  }

  ET_NODISCARD virtual runtime::Error update_encoder(
      const std::vector<runtime::EValue>& encoder_outputs) {
    (void)encoder_outputs;
    return runtime::Error::NotSupported;
  }

  /**
   * @brief The prefill and decode method names this IOManager drives, given
   * the methods the program has. Empty strings leave the runner's defaults.
   */
  virtual std::pair<std::string, std::string> method_names(
      const std::unordered_set<std::string>& available) const {
    (void)available;
    return {"", ""};
  }

 private:
  /**
   * @brief Reference to the Module used for method metadata and execution.
   */
  ET_MODULE_NAMESPACE::Module& module_;
};

using IOManagerFactory =
    std::function<std::unique_ptr<IOManager>(ET_MODULE_NAMESPACE::Module&)>;

/**
 * @brief IOManagers by the I/O contract they speak.
 *
 * A program names its contract in the `get_io_contract` constant method; the
 * runner builds the IOManager registered for it, or the default IOManager when
 * the program names none.
 */
inline std::unordered_map<std::string, IOManagerFactory>&
io_manager_registry() {
  static std::unordered_map<std::string, IOManagerFactory> registry;
  return registry;
}

inline bool register_io_manager(
    const std::string& contract,
    IOManagerFactory factory) {
  return io_manager_registry().emplace(contract, std::move(factory)).second;
}

inline runtime::Result<std::unique_ptr<IOManager>> make_io_manager(
    ET_MODULE_NAMESPACE::Module& module,
    const std::string& contract) {
  if (!contract.empty()) {
    auto it = io_manager_registry().find(contract);
    if (it != io_manager_registry().end()) {
      auto manager = it->second(module);
      if (!manager) {
        return runtime::Error::InvalidState;
      }
      return manager;
    }
    ET_LOG(
        Error,
        "No IOManager registered for I/O contract '%s'.",
        contract.c_str());
    return runtime::Error::NotSupported;
  }
  return std::make_unique<IOManager>(module);
}

/// Registers the IOManagers this library implements. Idempotent.
void register_builtin_io_managers();

/**
 * @brief Memory for the I/O buffers an IOManager owns.
 *
 * A backend whose device reads graph I/O from memory it can map (for example
 * QNN's rpcmem) registers an allocator, so the device uses these buffers in
 * place instead of copying them on every call. Without one, buffers come from
 * the heap.
 */
class ET_EXPERIMENTAL IOBufferAllocator {
 public:
  virtual ~IOBufferAllocator() = default;
  /// A block of at least `nbytes`, or nullptr to fall back to the heap.
  virtual void* allocate(size_t nbytes, size_t alignment) = 0;
  virtual void deallocate(void* block) = 0;
  /// Declares a tensor whose data starts at `tensor_data` inside `block`.
  virtual void bind(void* block, void* tensor_data) {
    (void)block;
    (void)tensor_data;
  }
};

using IOBufferAllocatorFactory =
    std::function<std::unique_ptr<IOBufferAllocator>()>;

inline IOBufferAllocatorFactory& io_buffer_allocator_factory() {
  static IOBufferAllocatorFactory factory;
  return factory;
}

/// Sets the allocator IOManagers use for their I/O buffers. The last call wins.
inline bool set_io_buffer_allocator(IOBufferAllocatorFactory factory) {
  io_buffer_allocator_factory() = std::move(factory);
  return true;
}

} // namespace llm
} // namespace extension
} // namespace executorch
