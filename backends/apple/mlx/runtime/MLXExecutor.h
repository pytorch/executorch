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

#include <iostream>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

// =============================================================================
// Op Logging - Enable via CMake: -DET_MLX_ENABLE_OP_LOGGING=1
// =============================================================================
#ifndef ET_MLX_ENABLE_OP_LOGGING
#define ET_MLX_ENABLE_OP_LOGGING 0
#endif

// =============================================================================
// Constant Zero-Copy - Enable via CMake: -DET_MLX_ENABLE_CONSTANT_ZERO_COPY=1
// When enabled, attempts to load model constants (weights) using zero-copy
// on Apple Silicon's unified memory. Falls back to copying if zero-copy fails.
// Disable if you want predictable memory usage (always copies).
// =============================================================================
#ifndef ET_MLX_ENABLE_CONSTANT_ZERO_COPY
#define ET_MLX_ENABLE_CONSTANT_ZERO_COPY 1 // Enabled by default
#endif

namespace executorch {
namespace backends {
namespace mlx {

// Compile-time logging flag
constexpr bool kEnableOpLogging = ET_MLX_ENABLE_OP_LOGGING;

// Compile-time constant zero-copy flag
constexpr bool kEnableConstantZeroCopy = ET_MLX_ENABLE_CONSTANT_ZERO_COPY;

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

  // Logging context
  size_t current_op_idx{0};
  const char* current_op_name{nullptr};

  // Tensor ID range boundaries for O(1) type lookup (computed at bind time)
  uint32_t input_end{0};
  uint32_t output_end{0};
  uint32_t mutable_buffer_end{0};

  void bind(const MLXProgram& prog, const ConstantData& const_data) {
    program = &prog;
    constants = &const_data;
    tensors.assign(prog.num_non_constant_tensors, std::nullopt);
    values.assign(prog.num_non_constant_values, std::nullopt);

    // Compute tensor ID range boundaries for fast type lookup
    // ID assignment order: Constant -> Input -> Output -> MutableBuffer -> Temp
    uint32_t constant_end = prog.num_constant_tensors;
    input_end = constant_end + prog.num_input_tensors;
    output_end = input_end + prog.num_output_tensors;
    mutable_buffer_end = output_end + prog.num_mutable_buffer_tensors;
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
  // Logging helpers
  // --------------------------

  static inline const char* dtype_str(::mlx::core::Dtype dtype) {
    using namespace ::mlx::core;
    switch (dtype.val()) {
      case float32.val():
        return "f32";
      case float16.val():
        return "f16";
      case bfloat16.val():
        return "bf16";
      case int32.val():
        return "i32";
      case int64.val():
        return "i64";
      case int16.val():
        return "i16";
      case int8.val():
        return "i8";
      case uint32.val():
        return "u32";
      case uint8.val():
        return "u8";
      case bool_.val():
        return "bool";
      default:
        return "?";
    }
  }

  static inline std::string format_shape(const ::mlx::core::Shape& shape) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0)
        ss << ",";
      ss << shape[i];
    }
    ss << ")";
    return ss.str();
  }

  static inline std::string format_tensor_info(const Tensor& t) {
    std::ostringstream ss;
    ss << dtype_str(t.dtype());
    ss << "(";
    const auto& shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0)
        ss << ",";
      ss << shape[i];
    }
    ss << ")";
    return ss.str();
  }

  // Get tensor type prefix for logging: "c", "i", "o", "b", "t"
  inline const char* tensor_type_prefix(Tid id) const {
    if (!program)
      return "?";

    uint32_t tid = id.idx;

    // Check each range in order (mutually exclusive ranges)
    if (tid < program->num_constant_tensors)
      return "c"; // Constant
    if (tid < input_end)
      return "i"; // User Input
    if (tid < output_end)
      return "o"; // User Output
    if (tid < mutable_buffer_end)
      return "b"; // Mutable Buffer
    return "t"; // Temp
  }

  inline void begin_op(size_t idx, const char* name) {
    current_op_idx = idx;
    current_op_name = name;
    if constexpr (kEnableOpLogging) {
      std::cout << "[" << idx << "] " << name << std::endl;
    }
  }

  inline void end_op() {
    if constexpr (kEnableOpLogging) {
      std::cout << "----\n";
    }
  }

  // --------------------------
  // Tensor accessors
  // --------------------------

  inline Tensor& tensor_ref(Tid id) {
    if constexpr (kEnableOpLogging) {
      std::cout << "  ref  " << tensor_type_prefix(id) << id.idx << std::flush;
    }
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
    if constexpr (kEnableOpLogging) {
      std::cout << "  " << format_tensor_info(*opt) << "\n";
    }
    return *opt;
  }

  inline const Tensor& const_tensor_ref(Tid id) const {
    const bool is_const = program && program->is_constant_tensor(id);
    if constexpr (kEnableOpLogging) {
      std::cout << "  in   " << tensor_type_prefix(id) << id.idx << std::flush;
    }
    if (!program) {
      throw std::runtime_error("const_tensor_ref: Program not bound");
    }
    if (id.idx >= program->num_tensors()) {
      throw std::out_of_range("const_tensor_ref: id out of range");
    }

    const Tensor* t = nullptr;
    if (is_const) {
      if (!constants) {
        throw std::runtime_error("const_tensor_ref: constants not bound");
      }
      t = &constants->get(id);
    } else {
      const auto& opt = tensors[id.idx - program->num_constant_tensors];
      if (!opt) {
        throw std::runtime_error(
            "const_tensor_ref: uninitialized tensor idx=" +
            std::to_string(id.idx));
      }
      t = &*opt;
    }

    if constexpr (kEnableOpLogging) {
      std::cout << "  " << format_tensor_info(*t) << "\n";
    }
    return *t;
  }

  // Set a tensor output
  inline void set_tensor(Tid id, Tensor arr) {
    if constexpr (kEnableOpLogging) {
      std::cout << "  out  " << tensor_type_prefix(id) << id.idx << "  "
                << format_tensor_info(arr) << "\n";
    }
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
    if constexpr (kEnableOpLogging) {
      std::cout << "  ref  v" << id.idx << std::flush;
    }
    if (id.idx >= values.size()) {
      throw std::out_of_range("value_ref: id out of range");
    }
    auto& opt = values[id.idx];
    if (!opt) {
      throw std::runtime_error(
          "value_ref: uninitialized value idx=" + std::to_string(id.idx));
    }
    if constexpr (kEnableOpLogging) {
      std::cout << "  " << std::get<T>(*opt) << "\n";
    }
    return std::get<T>(*opt);
  }

  template <typename T>
  inline const T& const_value_ref(Vid<T> id) const {
    if constexpr (kEnableOpLogging) {
      std::cout << "  in   v" << id.idx << std::flush;
    }
    if (id.idx >= values.size()) {
      throw std::out_of_range("const_value_ref: id out of range");
    }
    const auto& opt = values[id.idx];
    if (!opt) {
      throw std::runtime_error(
          "const_value_ref: uninitialized value idx=" + std::to_string(id.idx));
    }
    if constexpr (kEnableOpLogging) {
      std::cout << "  " << std::get<T>(*opt) << "\n";
    }
    return std::get<T>(*opt);
  }

  template <typename T>
  inline void set_value(Vid<T> id, T val) {
    if constexpr (kEnableOpLogging) {
      std::cout << "  out  v" << id.idx << "  " << val << "\n";
    }
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
// Helper to safely clamp int64_t to int32_t range
// =============================================================================

inline int32_t clamp_to_int32(int64_t val64) {
  // Clamp to int32_t range to avoid overflow
  // INT64_MAX is commonly used to mean "slice to end" or similar semantics
  if (val64 >= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return std::numeric_limits<int32_t>::max();
  } else if (
      val64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::min())) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(val64);
}

// =============================================================================
// Helper to resolve int or Vid with overflow protection
// =============================================================================

inline int32_t resolve_int(
    const std::variant<int64_t, Vid<int32_t>>& v,
    const ExecutionState& st) {
  if (std::holds_alternative<int64_t>(v)) {
    return clamp_to_int32(std::get<int64_t>(v));
  }
  return st.const_value_ref<int32_t>(std::get<Vid<int32_t>>(v));
}

// =============================================================================
// Helper to resolve vector of ints or Vids with overflow protection
// =============================================================================

inline std::vector<int32_t> resolve_ints(
    const std::vector<std::variant<int64_t, Vid<int32_t>>>& v,
    const ExecutionState& st) {
  std::vector<int32_t> out;
  out.reserve(v.size());
  for (const auto& elem : v) {
    out.push_back(resolve_int(elem, st));
  }
  return out;
}

// =============================================================================
// Helper to convert shape with potential dynamic dims
// =============================================================================

inline ::mlx::core::Shape to_shape(
    const std::vector<std::variant<int64_t, Vid<int32_t>>>& dims,
    const ExecutionState& st) {
  auto resolved = resolve_ints(dims, st);
  return ::mlx::core::Shape(resolved.begin(), resolved.end());
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
    out.push_back(clamp_to_int32(std::get<int64_t>(d)));
  }
  return out;
}

// =============================================================================
// Constant loading from raw bytes
// =============================================================================

// Load constants with zero-copy (when enabled)
// On Apple Silicon unified memory, MLX can wrap pointers directly
// If MLX falls back to copying, there may be temporary memory overhead (~2x)
// Helper struct to hold constant tensor metadata
struct ConstantTensorInfo {
  ::mlx::core::Shape shape;
  ::mlx::core::Dtype dtype;
  size_t nbytes;
  const void* data_ptr;
};

// Helper to compute constant tensor metadata and advance offset
inline ConstantTensorInfo get_constant_tensor_info(
    const MLXProgram& program,
    uint32_t tensor_id,
    const uint8_t* base,
    size_t& offset) {
  using namespace ::mlx::core;

  // Validate metadata exists
  if (tensor_id >= program.tensor_meta.size() ||
      !program.tensor_meta[tensor_id]) {
    throw std::runtime_error(
        "get_constant_tensor_info: missing metadata for constant " +
        std::to_string(tensor_id));
  }

  const auto& meta = *program.tensor_meta[tensor_id];
  Shape shape = to_shape(meta.shape);
  Dtype dtype = to_mlx_dtype(meta.dtype);

  // Align to 16 bytes
  offset = (offset + 15) & ~15ULL;

  // Calculate size
  size_t num_elements = 1;
  for (auto s : shape) {
    num_elements *= s;
  }
  size_t elem_size = size_of(dtype);
  size_t nbytes = num_elements * elem_size;

  const void* data_ptr = static_cast<const void*>(base + offset);

  return ConstantTensorInfo{shape, dtype, nbytes, data_ptr};
}

// Load constants with zero-copy by wrapping the constant buffer directly
inline void load_constants_zero_copy(
    const MLXProgram& program,
    ConstantData& store) {
  using namespace ::mlx::core;

  store.tensors.clear();

  if (program.num_constant_tensors == 0 || !program.constant_data) {
    return;
  }

  store.tensors.reserve(program.num_constant_tensors);

  const uint8_t* base = program.constant_data;
  size_t offset = 0;

  for (uint32_t tid = 0; tid < program.num_constant_tensors; ++tid) {
    ConstantTensorInfo info =
        get_constant_tensor_info(program, tid, base, offset);

    // Zero-copy: wrap pointer directly with no-op deleter
    void* data_ptr = const_cast<void*>(info.data_ptr);
    auto deleter = [](void*) {
      // Buffer will be freed when MLXHandle is destroyed
    };

    array arr = array(data_ptr, info.shape, info.dtype, deleter);
    store.add(std::move(arr));
    offset += info.nbytes;
  }
}

// Load constants with explicit copying
inline void load_constants_with_copy(
    const MLXProgram& program,
    ConstantData& store) {
  using namespace ::mlx::core;

  store.tensors.clear();

  if (program.num_constant_tensors == 0 || !program.constant_data) {
    return;
  }

  store.tensors.reserve(program.num_constant_tensors);

  const uint8_t* base = program.constant_data;
  size_t offset = 0;

  for (uint32_t tid = 0; tid < program.num_constant_tensors; ++tid) {
    ConstantTensorInfo info =
        get_constant_tensor_info(program, tid, base, offset);

    // Create array by copying data using typed constructor
    auto create_array = [&]() -> array {
      switch (info.dtype) {
        case float32:
          return array(
              static_cast<const float*>(info.data_ptr), info.shape, info.dtype);
        case float16:
          return array(
              static_cast<const float16_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case bfloat16:
          return array(
              static_cast<const bfloat16_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case int32:
          return array(
              static_cast<const int32_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case int64:
          return array(
              static_cast<const int64_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case int16:
          return array(
              static_cast<const int16_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case int8:
          return array(
              static_cast<const int8_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case uint32:
          return array(
              static_cast<const uint32_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case uint8:
          return array(
              static_cast<const uint8_t*>(info.data_ptr),
              info.shape,
              info.dtype);
        case bool_:
          return array(
              static_cast<const bool*>(info.data_ptr), info.shape, info.dtype);
        default:
          throw std::runtime_error(
              "load_constants_with_copy: unsupported dtype " +
              std::to_string(static_cast<int>(info.dtype.val())));
      }
    };

    store.add(create_array());
    offset += info.nbytes;
  }
}

// Public interface: dispatch based on compile-time flag
inline void load_constants(const MLXProgram& program, ConstantData& store) {
  if constexpr (kEnableConstantZeroCopy) {
    load_constants_zero_copy(program, store);
  } else {
    load_constants_with_copy(program, store);
  }

  // Evaluate all constants immediately to prepare Metal buffers
  // This trades init time for faster first inference
  using namespace ::mlx::core;
  eval(store.tensors);
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
