/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/nvidia/tensorrt/runtime/TensorRTBlobHeader.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace backends {
namespace tensorrt {

/**
 * I/O binding information for a TensorRT tensor.
 */
struct IOBinding {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  bool is_input;
  bool is_shape_tensor{false};
};

/**
 * GPU buffer information for pre-allocated memory.
 */
struct GPUBuffer {
  void* ptr{nullptr};
  size_t size{0};
  bool is_input{false};
  int32_t tensor_index{-1};
  size_t io_index{0}; // Index in input_buffers or output_buffers array
  bool has_dynamic_dims{false};
  bool is_shape_tensor{false};
};

/**
 * TensorRT executor for running inference with deserialized engines.
 *
 * This class wraps TensorRT runtime objects (engine, context) and provides
 * a simple interface for executing inference.
 *
 * Memory management patterns:
 * - TensorRT objects (IRuntime, ICudaEngine, IExecutionContext) use unique_ptr
 * - CUDA stream uses raw pointer with explicit cleanup
 * - GPU buffers are pre-allocated during initialize()
 * - On unified memory systems (Jetson), CPU-GPU copies are skipped
 */
class TensorRTExecutor {
 public:
  TensorRTExecutor() = default;
  ~TensorRTExecutor();

  TensorRTExecutor(const TensorRTExecutor&) = delete;
  TensorRTExecutor& operator=(const TensorRTExecutor&) = delete;
  TensorRTExecutor(TensorRTExecutor&&) noexcept;
  TensorRTExecutor& operator=(TensorRTExecutor&&) noexcept;

  /**
   * Initialize the executor with a serialized blob.
   *
   * Parses the blob header, deserializes the TensorRT engine, creates
   * an execution context, and pre-allocates GPU buffers.
   *
   * @param blob_data Pointer to the serialized blob.
   * @param blob_size Size of the blob in bytes.
   * @return Error::Ok on success.
   */
  runtime::Error initialize(const void* blob_data, size_t blob_size);

  /**
   * Execute inference with the given input/output buffers.
   *
   * On discrete GPUs: copies inputs to pre-allocated GPU memory, executes,
   * and copies outputs back. Input/output buffers may reside on either CPU
   * or GPU — the transfer direction is auto-detected via cudaMemcpyDefault.
   * On unified memory (Jetson): uses managed memory buffers directly.
   *
   * @param input_buffers Array of pointers to input data buffers (CPU or GPU).
   * @param num_inputs Number of input buffers.
   * @param output_buffers Array of pointers to output data buffers (CPU or
   * GPU).
   * @param num_outputs Number of output buffers.
   * @return Error::Ok on success.
   */
  runtime::Error execute(
      void* const* input_buffers,
      const std::vector<std::vector<int64_t>>& input_shapes,
      size_t num_inputs,
      void* const* output_buffers,
      size_t num_outputs);

  /**
   * Get the inferred output shape after the last execute() call.
   * Only meaningful when the engine has dynamic shapes.
   */
  const std::vector<int32_t>& get_output_shape(size_t output_index) const {
    return output_shapes_[output_index];
  }

  /**
   * Get I/O binding information.
   *
   * @return Vector of IOBinding structs describing inputs and outputs.
   */
  const std::vector<IOBinding>& get_io_bindings() const {
    return io_bindings_;
  }

  /**
   * Check if the executor is initialized.
   *
   * @return true if initialized with a valid engine.
   */
  bool is_initialized() const {
    return engine_ != nullptr;
  }

  /**
   * Get the number of input tensors.
   *
   * @return Number of input tensors in the engine.
   */
  size_t get_num_inputs() const;

  /**
   * Get the number of output tensors.
   *
   * @return Number of output tensors in the engine.
   */
  size_t get_num_outputs() const;

  /**
   * Check if a given input index corresponds to a shape tensor.
   * Shape tensors carry dimension values for dynamic shapes and
   * live on host memory rather than device memory.
   */
  bool is_input_shape_tensor(size_t input_index) const;

  /**
   * Get the element size in bytes for a TRT output tensor.
   * Used to detect dtype mismatches (e.g., TRT int32 vs ExecuTorch int64).
   */
  size_t get_output_dtype_size(size_t output_index) const;

  /**
   * Check if running on unified memory system (e.g., Jetson).
   *
   * @return true if CPU and GPU share memory.
   */
  bool uses_unified_memory() const {
    return uses_unified_memory_;
  }

  /**
   * Set an external CUDA stream for this executor.
   *
   * When called before initialize(), the executor uses the provided stream
   * instead of creating its own. This enables stream sharing across multiple
   * TRT delegate instances for serialized execution, avoiding synchronization
   * overhead between subgraphs.
   *
   * @param stream External CUDA stream to use.
   * @param owns_stream If true, the executor will destroy the stream on
   * cleanup. If false, the caller retains ownership.
   */
  void set_cuda_stream(::cudaStream_t stream, bool owns_stream = false);

 private:
  /**
   * Parse I/O binding metadata from JSON.
   *
   * @param json_data Pointer to JSON data.
   * @param json_size Size of JSON data in bytes.
   * @return true on success.
   */
  bool parse_io_bindings(const void* json_data, size_t json_size);

  /**
   * Pre-allocate GPU buffers for all I/O tensors.
   *
   * @return Error::Ok on success.
   */
  runtime::Error allocate_gpu_buffers();

  /**
   * Free all pre-allocated GPU buffers.
   */
  void free_gpu_buffers();

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  ::cudaStream_t stream_{nullptr};
  bool owns_stream_{true};
  std::vector<IOBinding> io_bindings_;
  std::vector<GPUBuffer> gpu_buffers_;
  std::vector<std::vector<int32_t>> shape_tensor_host_buffers_;
  bool uses_unified_memory_{false};
  bool has_dynamic_shapes_{false};
  std::vector<std::vector<int32_t>> output_shapes_;
};

} // namespace tensorrt
} // namespace backends
} // namespace executorch
