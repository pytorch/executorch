/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tag.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace executorch_flatbuffer {
struct ExecutionPlan;
} // namespace executorch_flatbuffer

namespace executorch {
namespace runtime {

/**
 * Metadata about a specific tensor of an ExecuTorch Program.
 *
 * The program used to create the MethodMeta object that created this
 * TensorInfo must outlive this TensorInfo.
 */
class TensorInfo final {
 public:
  TensorInfo() = delete;
  TensorInfo(const TensorInfo&) = default;
  TensorInfo(TensorInfo&&) = default;
  TensorInfo& operator=(const TensorInfo&) = default;
  TensorInfo& operator=(TensorInfo&& other) = default;
  ~TensorInfo() = default;

  /**
   * Returns the sizes of the tensor.
   */
  Span<const int32_t> sizes() const;

  /**
   * Returns the dim order of the tensor.
   */
  Span<const uint8_t> dim_order() const;

  /**
   * Returns the scalar type of the input/output.
   */
  executorch::aten::ScalarType scalar_type() const;

  /**
   * Returns whether the tensor's memory was planned during export.
   */
  bool is_memory_planned() const;

  /**
   * Returns the size of the tensor in bytes.
   */
  size_t nbytes() const;

 private:
  // Let MethodMeta create TensorInfo.
  friend class MethodMeta;

  TensorInfo(
      Span<const int32_t> sizes,
      Span<const uint8_t> dim_order,
      executorch::aten::ScalarType scalar_type,
      const bool is_memory_planned);

  /**
   * The sizes of the tensor.
   *
   * NOTE: References data from the Program, so the Program must outlive the
   * TensorInfo.
   */
  Span<const int32_t> sizes_;

  /**
   * The dim order of the tensor.
   *
   * NOTE: References data from the Program, so the Program must outlive the
   * TensorInfo.
   */
  Span<const uint8_t> dim_order_;

  /// The scalar type of the tensor.
  executorch::aten::ScalarType scalar_type_;

  /// Whether the tensor's memory was planned during export.
  bool is_memory_planned_;

  /// The size in bytes of the tensor.
  size_t nbytes_;
};

/**
 * Describes a a method in an ExecuTorch program.
 *
 * The program used to create a MethodMeta object must outlive the MethodMeta.
 * It is separate from Method so that this information can be accessed without
 * paying the initialization cost of loading the full Method.
 */
class MethodMeta final {
 public:
  MethodMeta() = delete;
  MethodMeta(const MethodMeta&) = default;
  MethodMeta(MethodMeta&&) = default;
  MethodMeta& operator=(const MethodMeta&) = default;
  MethodMeta& operator=(MethodMeta&& other) = default;
  ~MethodMeta() = default;

  /**
   * Get the name of this method.
   *
   * @returns The method name.
   */
  const char* name() const;

  /**
   * Get the number of inputs to this method.
   *
   * @returns The number of inputs.
   */
  size_t num_inputs() const;

  /**
   * Get the tag of the specified input.
   *
   * @param[in] index The index of the input to look up.
   * @returns The tag of input, can only be [Tensor, Int, Bool, Double, String].
   */
  Result<Tag> input_tag(size_t index) const;

  /**
   * Get metadata about the specified input.
   *
   * @param[in] index The index of the input to look up.
   * @returns The metadata on success, or an error on failure. Only valid for
   * tag::Tensor
   */
  Result<TensorInfo> input_tensor_meta(size_t index) const;

  /**
   * Get the number of outputs to this method.
   *
   * @returns The number of outputs.
   */
  size_t num_outputs() const;

  /**
   * Get the tag of the specified output.
   *
   * @param[in] index The index of the output to look up.
   * @returns The tag of output, can only be [Tensor, Int, Bool, Double,
   * String].
   */
  Result<Tag> output_tag(size_t index) const;

  /**
   * Get metadata about the specified output.
   *
   * @param[in] index The index of the output to look up.
   * @returns The metadata on success, or an error on failure. Only valid for
   * tag::Tensor
   */
  Result<TensorInfo> output_tensor_meta(size_t index) const;

  /**
   * Get the number of memory-planned buffers this method requires.
   *
   * @returns The number of memory-planned buffers.
   */
  size_t num_memory_planned_buffers() const;

  /**
   * Get the size in bytes of the specified memory-planned buffer.
   *
   * @param[in] index The index of the buffer to look up.
   * @returns The size in bytes on success, or an error on failure.
   */
  Result<int64_t> memory_planned_buffer_size(size_t index) const;

  /**
   * DEPRECATED: Use num_memory_planned_buffers() instead.
   */
  ET_DEPRECATED size_t num_non_const_buffers() const {
    return num_memory_planned_buffers();
  }

  /**
   * DEPRECATED: Use memory_planned_buffer_size() instead.
   */
  Result<int64_t> non_const_buffer_size(size_t index) const {
    return memory_planned_buffer_size(index);
  }

 private:
  // Let Program create MethodMeta.
  friend class Program;

  explicit MethodMeta(const executorch_flatbuffer::ExecutionPlan* s_plan);

  /// Source of truth for method information
  const executorch_flatbuffer::ExecutionPlan* s_plan_;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::TensorInfo;
} // namespace executor
} // namespace torch
