//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#pragma once

#include "MLXLoader.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include <executorch/runtime/core/error.h>

#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace mlx {

// =============================================================================
// Type aliases
// =============================================================================

using Tensor = ::mlx::core::array;
using Value = std::variant<int32_t, float, bool>;
using StreamOrDevice = ::mlx::core::StreamOrDevice;

// =============================================================================
// ConstantData - storage for loaded constants
// =============================================================================

struct ConstantData {
  std::vector<Tensor> tensors;

  inline const Tensor& get(Tid id) const {
    if (id.idx >= tensors.size()) {
      throw std::out_of_range("ConstantData::get: id out of range");
    }
    return tensors[id.idx];
  }

  inline void add(Tensor t) {
    tensors.push_back(std::move(t));
  }
};

// =============================================================================
// MutableBufferData - storage for delegate-owned mutable buffers (e.g., KV
// cache) These persist across execute() calls, avoiding per-execution copies.
// =============================================================================

struct MutableBufferData {
  // Maps tensor slot idx to MLX array
  // Using vector of optional since mlx::array has no default constructor
  std::vector<std::optional<Tensor>> tensors;

  inline void resize(size_t n) {
    tensors.resize(n, std::nullopt);
  }

  inline bool has(Tid id) const {
    return id.idx < tensors.size() && tensors[id.idx].has_value();
  }

  inline Tensor& get(Tid id) {
    if (id.idx >= tensors.size() || !tensors[id.idx].has_value()) {
      throw std::out_of_range("MutableBufferData::get: id not found or unset");
    }
    return *tensors[id.idx];
  }

  inline const Tensor& get(Tid id) const {
    if (id.idx >= tensors.size() || !tensors[id.idx].has_value()) {
      throw std::out_of_range("MutableBufferData::get: id not found or unset");
    }
    return *tensors[id.idx];
  }

  inline void set(Tid id, Tensor t) {
    if (id.idx >= tensors.size()) {
      tensors.resize(id.idx + 1, std::nullopt);
    }
    tensors[id.idx] = std::move(t);
  }

  inline void clear() {
    tensors.clear();
  }
};

// =============================================================================
// ExecutionState - per-run mutable state
// =============================================================================

struct ExecutionState {
  const MLXProgram* program{nullptr};
  const ConstantData* constants{nullptr};

  // Non-constant tensors (inputs, outputs, mutable buffers, temps)
  std::vector<std::optional<Tensor>> tensors;

  // Non-constant values (SymInt, etc.)
  std::vector<std::optional<Value>> values;

  void bind(const MLXProgram& prog, const ConstantData& const_data) {
    program = &prog;
    constants = &const_data;
    tensors.assign(prog.num_non_constant_tensors, std::nullopt);
    values.assign(prog.num_non_constant_values, std::nullopt);
  }

  void reset() {
    // Clear non-constant tensors/values for reuse
    for (auto& t : tensors) {
      t = std::nullopt;
    }
    for (auto& v : values) {
      v = std::nullopt;
    }
  }

  // --------------------------
  // Tensor accessors
  // --------------------------

  inline Tensor& tensor_ref(Tid id) {
    if (!program) {
      throw std::runtime_error("tensor_ref: Program not bound");
    }
    if (id.idx >= program->num_tensors()) {
      throw std::out_of_range("tensor_ref: id out of range");
    }
    if (program->is_constant_tensor(id)) {
      throw std::runtime_error("tensor_ref: cannot mutate constant tensor");
    }
    auto& opt = tensors[id.idx - program->num_constant_tensors];
    if (!opt) {
      throw std::runtime_error(
          "tensor_ref: uninitialized tensor idx=" + std::to_string(id.idx));
    }
    return *opt;
  }

  inline const Tensor& const_tensor_ref(Tid id) const {
    if (!program) {
      throw std::runtime_error("const_tensor_ref: Program not bound");
    }
    if (id.idx >= program->num_tensors()) {
      throw std::out_of_range("const_tensor_ref: id out of range");
    }

    if (program->is_constant_tensor(id)) {
      if (!constants) {
        throw std::runtime_error("const_tensor_ref: constants not bound");
      }
      return constants->get(id);
    }

    const auto& opt = tensors[id.idx - program->num_constant_tensors];
    if (!opt) {
      throw std::runtime_error(
          "const_tensor_ref: uninitialized tensor idx=" +
          std::to_string(id.idx));
    }
    return *opt;
  }

  // Set a tensor output
  inline void set_tensor(Tid id, Tensor arr) {
    if (!program) {
      throw std::runtime_error("set_tensor: Program not bound");
    }
    if (id.idx < program->num_constant_tensors) {
      throw std::runtime_error("set_tensor: cannot write to constant tensor");
    }
    uint32_t off = id.idx - program->num_constant_tensors;
    if (off >= tensors.size()) {
      throw std::out_of_range("set_tensor: tensor idx out of range");
    }
    tensors[off] = std::move(arr);
  }

  // --------------------------
  // Value accessors
  // --------------------------

  template <typename T>
  inline T& value_ref(Vid<T> id) {
    if (id.idx >= values.size()) {
      throw std::out_of_range("value_ref: id out of range");
    }
    auto& opt = values[id.idx];
    if (!opt) {
      throw std::runtime_error(
          "value_ref: uninitialized value idx=" + std::to_string(id.idx));
    }
    return std::get<T>(*opt);
  }

  template <typename T>
  inline const T& const_value_ref(Vid<T> id) const {
    if (id.idx >= values.size()) {
      throw std::out_of_range("const_value_ref: id out of range");
    }
    const auto& opt = values[id.idx];
    if (!opt) {
      throw std::runtime_error(
          "const_value_ref: uninitialized value idx=" + std::to_string(id.idx));
    }
    return std::get<T>(*opt);
  }

  template <typename T>
  inline void set_value(Vid<T> id, T val) {
    if (id.idx >= values.size()) {
      throw std::out_of_range("set_value: id out of range");
    }
    values[id.idx] = val;
  }
};

// =============================================================================
// Dtype conversion
// =============================================================================

inline ::mlx::core::Dtype to_mlx_dtype(DTypeId d) {
  using namespace ::mlx::core;
  switch (d) {
    case DTypeId::f16:
      return float16;
    case DTypeId::f32:
      return float32;
    case DTypeId::bf16:
      return bfloat16;
    case DTypeId::i32:
      return int32;
    case DTypeId::i64:
      return int64;
    case DTypeId::u32:
      return uint32;
    case DTypeId::u8:
      return uint8;
    case DTypeId::boolean:
      return bool_;
    case DTypeId::i8:
      return int8;
    default:
      return float32;
  }
}

// =============================================================================
// Helper to convert shape with potential dynamic dims
// =============================================================================

inline ::mlx::core::Shape to_shape(
    const std::vector<std::variant<int64_t, Vid<int32_t>>>& dims,
    const ExecutionState& st) {
  ::mlx::core::Shape out;
  out.reserve(dims.size());
  for (const auto& d : dims) {
    if (std::holds_alternative<int64_t>(d)) {
      out.push_back(static_cast<int32_t>(std::get<int64_t>(d)));
    } else {
      int32_t v = st.const_value_ref<int32_t>(std::get<Vid<int32_t>>(d));
      out.push_back(v);
    }
  }
  return out;
}

inline ::mlx::core::Shape to_shape(const std::vector<int32_t>& dims) {
  return ::mlx::core::Shape(dims.begin(), dims.end());
}

// Overload for static shapes (used when loading constants where all dims must
// be literals)
inline ::mlx::core::Shape to_shape(
    const std::vector<std::variant<int64_t, Vid<int32_t>>>& dims) {
  ::mlx::core::Shape out;
  out.reserve(dims.size());
  for (const auto& d : dims) {
    if (!std::holds_alternative<int64_t>(d)) {
      throw std::runtime_error(
          "to_shape: expected static shape but found dynamic Vid reference");
    }
    out.push_back(static_cast<int32_t>(std::get<int64_t>(d)));
  }
  return out;
}

// =============================================================================
// Constant loading from raw bytes
// =============================================================================

inline void load_constants(const MLXProgram& program, ConstantData& store) {
  using namespace ::mlx::core;

  store.tensors.clear();

  if (program.num_constant_tensors == 0 || !program.constant_data) {
    return;
  }

  store.tensors.reserve(program.num_constant_tensors);

  const uint8_t* base = program.constant_data;
  size_t offset = 0;

  for (uint32_t tid = 0; tid < program.num_constant_tensors; ++tid) {
    // Get metadata
    if (tid >= program.tensor_meta.size() || !program.tensor_meta[tid]) {
      throw std::runtime_error(
          "load_constants: missing metadata for constant " +
          std::to_string(tid));
    }

    const auto& meta = *program.tensor_meta[tid];
    auto shape = to_shape(meta.shape);
    auto dtype = to_mlx_dtype(meta.dtype);

    // Align to 16 bytes
    offset = (offset + 15) & ~15ULL;

    // Calculate size
    size_t num_elements = 1;
    for (auto s : shape) {
      num_elements *= s;
    }
    size_t elem_size = size_of(dtype);
    size_t nbytes = num_elements * elem_size;

    // Create array by copying data from CPU pointer
    // MLX requires proper Metal-aligned memory, so we copy the data
    const void* src_ptr = static_cast<const void*>(base + offset);

    // Helper lambda to create the array with proper typed constructor
    auto create_array = [&]() -> array {
      switch (dtype) {
        case float32:
          return array(static_cast<const float*>(src_ptr), shape, dtype);
        case float16:
          return array(static_cast<const float16_t*>(src_ptr), shape, dtype);
        case bfloat16:
          return array(static_cast<const bfloat16_t*>(src_ptr), shape, dtype);
        case int32:
          return array(static_cast<const int32_t*>(src_ptr), shape, dtype);
        case int64:
          return array(static_cast<const int64_t*>(src_ptr), shape, dtype);
        case int16:
          return array(static_cast<const int16_t*>(src_ptr), shape, dtype);
        case int8:
          return array(static_cast<const int8_t*>(src_ptr), shape, dtype);
        case uint32:
          return array(static_cast<const uint32_t*>(src_ptr), shape, dtype);
        case uint8:
          return array(static_cast<const uint8_t*>(src_ptr), shape, dtype);
        case bool_:
          return array(static_cast<const bool*>(src_ptr), shape, dtype);
        default:
          throw std::runtime_error(
              "load_constants: unsupported dtype " +
              std::to_string(static_cast<int>(dtype.val())));
      }
    };

    store.add(create_array());
    offset += nbytes;
  }
}

// =============================================================================
// Mutable buffer initialization
// Creates MLX arrays for mutable buffers (e.g., KV cache) at init time.
// These persist in GPU memory across execute() calls.
// =============================================================================

inline void load_mutable_buffers(
    const MLXProgram& program,
    MutableBufferData& store) {
  using namespace ::mlx::core;

  store.clear();

  if (program.mutable_buffer_map.empty()) {
    return;
  }

  for (const auto& slot : program.mutable_buffer_map) {
    if (slot.slot_type != SlotType::TensorSlot) {
      continue;
    }

    Tid tid{slot.idx};

    // Get metadata for this tensor
    if (tid.idx >= program.tensor_meta.size()) {
      ET_LOG(
          Error,
          "load_mutable_buffers: tid %u >= tensor_meta.size() %zu",
          tid.idx,
          program.tensor_meta.size());
      throw std::runtime_error(
          "load_mutable_buffers: tensor index out of range for tensor " +
          std::to_string(tid.idx));
    }

    if (!program.tensor_meta[tid.idx]) {
      ET_LOG(
          Error,
          "load_mutable_buffers: missing metadata for tensor %u",
          tid.idx);
      throw std::runtime_error(
          "load_mutable_buffers: missing metadata for tensor " +
          std::to_string(tid.idx));
    }

    const auto& meta = *program.tensor_meta[tid.idx];
    auto shape = to_shape(meta.shape);
    auto dtype = to_mlx_dtype(meta.dtype);

    // Initialize mutable buffer to zeros
    // This matches the typical initialization of KV cache buffers
    auto arr = zeros(shape, dtype);

    // Evaluate immediately to allocate in GPU memory
    eval(arr);

    store.set(tid, std::move(arr));
  }
}

} // namespace mlx
} // namespace backends
} // namespace executorch
