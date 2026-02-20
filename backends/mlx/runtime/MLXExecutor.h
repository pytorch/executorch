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
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

// =============================================================================
// Op Logging - compile-time gate + runtime env var check
//
// Compile flag (CMake: -DET_MLX_ENABLE_OP_LOGGING=1) controls whether logging
// code is compiled in at all. When off, all logging is stripped (zero
// overhead). When on, the env var ET_MLX_ENABLE_OP_LOGGING=1 must also be set
// at runtime to actually produce output.
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

// Runtime check for op logging (only callable when compiled in)
#if ET_MLX_ENABLE_OP_LOGGING
inline bool isOpLoggingEnabled() {
  static const bool enabled = []() {
    const char* val = std::getenv("ET_MLX_ENABLE_OP_LOGGING");
    return val != nullptr && std::string(val) == "1";
  }();
  return enabled;
}
#else
constexpr bool isOpLoggingEnabled() {
  return false;
}
#endif

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
      throw std::out_of_range("MutableBufferData::set: id out of range");
    }
    tensors[id.idx] = std::move(t);
  }

  inline void clear() {
    tensors.clear();
  }
};

// =============================================================================
// ExecutionState - reusable execution context
//
// Design: Separates tensors by lifetime for clarity and shareability:
// - Constants: stored in ConstantData, accessed via pointer (can be shared)
// - Mutable buffers: stored in MutableBufferData, accessed via pointer
// (persistent)
// - Inputs/outputs/temps: stored in tensors vector (per-execution)
// =============================================================================

struct ExecutionState {
  const MLXProgram* program{nullptr};
  const ConstantData* constants{nullptr}; // Shared, read-only
  MutableBufferData* mutable_buffers{nullptr}; // Per-handle, persistent

  // Per-execution tensors: inputs, outputs, temps (NOT constants or mutable
  // buffers)
  std::vector<std::optional<Tensor>> tensors;

  // Non-constant values (SymInt, etc.)
  std::vector<std::optional<Value>> values;

  // Logging context
  size_t current_op_idx{0};
  const char* current_op_name{nullptr};

  // Tensor ID range boundaries for O(1) type lookup (computed at bind time)
  uint32_t num_constants{0};
  uint32_t input_end{0};
  uint32_t output_end{0};
  uint32_t mutable_buffer_end{0};

  void bind(
      const MLXProgram& prog,
      const ConstantData& const_data,
      MutableBufferData& mut_bufs) {
    program = &prog;
    constants = &const_data;
    mutable_buffers = &mut_bufs;

    // Allocate space for inputs, outputs, and temps only (not constants or
    // mutable buffers)
    size_t num_per_execution_tensors = prog.num_input_tensors +
        prog.num_output_tensors + prog.num_temp_tensors;
    tensors.assign(num_per_execution_tensors, std::nullopt);
    values.assign(prog.num_values, std::nullopt);

    // Compute tensor ID range boundaries for fast type lookup
    // ID assignment order: Constant -> Input -> Output -> MutableBuffer -> Temp
    num_constants = prog.num_constant_tensors;
    input_end = num_constants + prog.num_input_tensors;
    output_end = input_end + prog.num_output_tensors;
    mutable_buffer_end = output_end + prog.num_mutable_buffer_tensors;
  }

  // Check if a tensor ID is a mutable buffer
  inline bool is_mutable_buffer(Tid id) const {
    return id.idx >= output_end && id.idx < mutable_buffer_end;
  }

  // Convert tensor ID to index in the tensors vector
  // Accounts for constants and mutable buffers not being in the vector
  inline uint32_t tensor_index(Tid id) const {
    if (id.idx < num_constants) {
      throw std::runtime_error(
          "tensor_index: called with constant tensor id " +
          std::to_string(id.idx));
    }
    uint32_t idx = id.idx - num_constants;
    // If this ID is after mutable buffer range, subtract mutable buffer count
    if (id.idx >= mutable_buffer_end) {
      idx -= program->num_mutable_buffer_tensors;
    }
    return idx;
  }

  void reset() {
    // Clear per-execution tensors (inputs, outputs, temps)
    // Constants and mutable buffers are not in this vector
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

  // Compute tensor stats: min, max, mean, nan_count
  // Uses MLX ops for GPU-accelerated computation
  static inline std::string format_tensor_stats(const Tensor& t) {
    using namespace ::mlx::core;

    try {
      std::ostringstream ss;

      size_t numel = t.size();
      if (numel == 0) {
        ss << "[empty]";
        return ss.str();
      }

      // Cast to float32 for stats computation (handles bf16/fp16/int/bool)
      Tensor t_float = astype(t, float32);

      // Use MLX ops for efficient GPU-based stats
      Tensor nan_mask = isnan(t_float);
      Tensor inf_mask = isinf(t_float);
      Tensor nan_count_arr = sum(astype(nan_mask, int32));
      Tensor inf_count_arr = sum(astype(inf_mask, int32));

      // For min/max/mean, we need to handle NaN/Inf - replace with 0
      Tensor valid_mask = logical_not(logical_or(nan_mask, inf_mask));
      Tensor t_valid = where(valid_mask, t_float, zeros_like(t_float));

      Tensor min_arr = min(t_valid);
      Tensor max_arr = max(t_valid);
      Tensor mean_arr = mean(t_valid);

      // Evaluate all at once
      eval({nan_count_arr, inf_count_arr, min_arr, max_arr, mean_arr});

      int nan_count = nan_count_arr.item<int>();
      int inf_count = inf_count_arr.item<int>();
      float min_val = min_arr.item<float>();
      float max_val = max_arr.item<float>();
      float mean_val = mean_arr.item<float>();

      ss << std::fixed << std::setprecision(4);
      ss << "[min=" << min_val << " max=" << max_val << " mean=" << mean_val;
      if (nan_count > 0) {
        ss << " NaN=" << nan_count;
      }
      if (inf_count > 0) {
        ss << " Inf=" << inf_count;
      }
      ss << "]";
      return ss.str();
    } catch (const std::exception& e) {
      return std::string("[stats error: ") + e.what() + "]";
    } catch (...) {
      return "[stats error: unknown]";
    }
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
    if (isOpLoggingEnabled()) {
      std::cout << "[" << idx << "] " << name << std::endl;
    }
  }

  inline void end_op() {
    if (isOpLoggingEnabled()) {
      std::cout << "----\n";
    }
  }

  // --------------------------
  // Tensor accessors
  // --------------------------

  inline Tensor& tensor_ref(Tid id) {
    if (isOpLoggingEnabled()) {
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
    // Route to mutable buffers or per-execution tensors
    Tensor* t = nullptr;
    if (is_mutable_buffer(id)) {
      if (!mutable_buffers) {
        throw std::runtime_error("tensor_ref: mutable_buffers not bound");
      }
      t = &mutable_buffers->get(id);
    } else {
      uint32_t idx = tensor_index(id);
      if (idx >= tensors.size()) {
        throw std::out_of_range("tensor_ref: tensor idx out of range");
      }
      auto& opt = tensors[idx];
      if (!opt) {
        throw std::runtime_error(
            "tensor_ref: uninitialized tensor idx=" + std::to_string(id.idx));
      }
      t = &*opt;
    }
    if (isOpLoggingEnabled()) {
      std::cout << "  " << format_tensor_info(*t) << "\n";
    }
    return *t;
  }

  inline const Tensor& const_tensor_ref(Tid id) const {
    if (isOpLoggingEnabled()) {
      std::cout << "  in   " << tensor_type_prefix(id) << id.idx << std::flush;
    }
    if (!program) {
      throw std::runtime_error("const_tensor_ref: Program not bound");
    }
    if (id.idx >= program->num_tensors()) {
      throw std::out_of_range("const_tensor_ref: id out of range");
    }

    const Tensor* t = nullptr;
    if (program->is_constant_tensor(id)) {
      // Route to constants
      if (!constants) {
        throw std::runtime_error("const_tensor_ref: constants not bound");
      }
      t = &constants->get(id);
    } else if (is_mutable_buffer(id)) {
      // Route to mutable buffers
      if (!mutable_buffers) {
        throw std::runtime_error("const_tensor_ref: mutable_buffers not bound");
      }
      t = &mutable_buffers->get(id);
    } else {
      // Route to per-execution tensors
      uint32_t idx = tensor_index(id);
      if (idx >= tensors.size()) {
        throw std::out_of_range("const_tensor_ref: tensor idx out of range");
      }
      const auto& opt = tensors[idx];
      if (!opt) {
        throw std::runtime_error(
            "const_tensor_ref: uninitialized tensor idx=" +
            std::to_string(id.idx));
      }
      t = &*opt;
    }

    if (isOpLoggingEnabled()) {
      std::cout << "  " << format_tensor_info(*t) << " "
                << format_tensor_stats(*t) << "\n";
    }
    return *t;
  }

  // Set a tensor output
  inline void set_tensor(Tid id, Tensor arr) {
    if (isOpLoggingEnabled()) {
      std::cout << "  out  " << tensor_type_prefix(id) << id.idx << "  "
                << format_tensor_info(arr) << " " << format_tensor_stats(arr)
                << "\n";
    }
    if (!program) {
      throw std::runtime_error("set_tensor: Program not bound");
    }
    if (id.idx < program->num_constant_tensors) {
      throw std::runtime_error("set_tensor: cannot write to constant tensor");
    }
    // Route to mutable buffers or per-execution tensors
    if (is_mutable_buffer(id)) {
      if (!mutable_buffers) {
        throw std::runtime_error("set_tensor: mutable_buffers not bound");
      }
      mutable_buffers->set(id, std::move(arr));
    } else {
      uint32_t idx = tensor_index(id);
      if (idx >= tensors.size()) {
        throw std::out_of_range("set_tensor: tensor idx out of range");
      }
      tensors[idx] = std::move(arr);
    }
  }

  // --------------------------
  // Value accessors
  // --------------------------

  template <typename T>
  inline T& value_ref(Vid id) {
    if (isOpLoggingEnabled()) {
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
    if (isOpLoggingEnabled()) {
      std::cout << "  " << std::get<T>(*opt) << "\n";
    }
    return std::get<T>(*opt);
  }

  template <typename T>
  inline const T& const_value_ref(Vid id) const {
    if (isOpLoggingEnabled()) {
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
    if (isOpLoggingEnabled()) {
      std::cout << "  " << std::get<T>(*opt) << "\n";
    }
    return std::get<T>(*opt);
  }

  inline const Value& const_value(Vid id) const {
    if (isOpLoggingEnabled()) {
      std::cout << "  in   v" << id.idx << std::flush;
    }
    if (id.idx >= values.size()) {
      throw std::out_of_range("const_value: id out of range");
    }
    const auto& opt = values[id.idx];
    if (!opt) {
      throw std::runtime_error(
          "const_value: uninitialized value idx=" + std::to_string(id.idx));
    }
    if (isOpLoggingEnabled()) {
      std::visit([](auto&& arg) { std::cout << "  " << arg << "\n"; }, *opt);
    }
    return *opt;
  }

  template <typename T>
  inline void set_value(Vid id, T val) {
    if (isOpLoggingEnabled()) {
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

inline ::mlx::core::Dtype resolve_dtype(ScalarType d) {
  using namespace ::mlx::core;
  switch (d) {
    case ScalarType::Half:
      return float16;
    case ScalarType::Float:
      return float32;
    case ScalarType::BFloat16:
      return bfloat16;
    case ScalarType::Int:
      return int32;
    case ScalarType::Short:
      return int16;
    case ScalarType::Long:
      return int64;
    case ScalarType::UInt32:
      return uint32;
    case ScalarType::Byte:
      return uint8;
    case ScalarType::Bool:
      return bool_;
    case ScalarType::Char:
      return int8;
    default:
      throw std::runtime_error(
          "Unsupported ScalarType: " + std::to_string(static_cast<int>(d)));
  }
}

inline ::mlx::core::Dtype resolve_dtype(int8_t d) {
  return resolve_dtype(static_cast<ScalarType>(d));
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
    const std::variant<int64_t, Vid>& v,
    const ExecutionState& st) {
  if (std::holds_alternative<int64_t>(v)) {
    return clamp_to_int32(std::get<int64_t>(v));
  }
  return st.const_value_ref<int32_t>(std::get<Vid>(v));
}

// =============================================================================
// Helper to resolve vector of ints or Vids with overflow protection
// =============================================================================

inline std::vector<int32_t> resolve_ints(
    const std::vector<std::variant<int64_t, Vid>>& v,
    const ExecutionState& st) {
  std::vector<int32_t> out;
  out.reserve(v.size());
  for (const auto& elem : v) {
    out.push_back(resolve_int(elem, st));
  }
  return out;
}

// =============================================================================
// Helper to resolve float or Vid
// =============================================================================

inline float resolve_float(
    const std::variant<double, Vid>& v,
    const ExecutionState& st) {
  if (std::holds_alternative<double>(v)) {
    return static_cast<float>(std::get<double>(v));
  }
  // The value may be stored as int32_t (from SymInt computations) or float.
  const auto& val = st.const_value(std::get<Vid>(v));
  return std::visit(
      [](auto&& arg) -> float { return static_cast<float>(arg); }, val);
}

// =============================================================================
// Helper to convert shape with potential dynamic dims
// =============================================================================

inline ::mlx::core::Shape to_shape(
    const std::vector<std::variant<int64_t, Vid>>& dims,
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
    const std::vector<std::variant<int64_t, Vid>>& dims) {
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
// Constant loading from NamedDataMap
// =============================================================================

// Load constants from ExecuTorch's NamedDataMap.
// Constants are stored by name in the .pte file and loaded via the
// named_data_map interface. This allows ExecuTorch to own the constant data and
// enables zero-copy on Apple Silicon unified memory.
//
// Parameters:
//   program: The loaded MLXProgram containing tensor metadata and named_slots
//   named_data_map: ExecuTorch's interface for accessing named data
//   store: Output storage for loaded constant tensors
//   constant_buffers: Vector to store FreeableBuffers (must outlive store for
//   zero-copy)
inline void load_constants(
    const MLXProgram& program,
    const runtime::NamedDataMap* named_data_map,
    ConstantData& store,
    std::vector<runtime::FreeableBuffer>& constant_buffers) {
  using namespace ::mlx::core;

  store.tensors.clear();
  constant_buffers.clear();

  if (program.num_constant_tensors == 0) {
    return;
  }

  store.tensors.reserve(program.num_constant_tensors);
  constant_buffers.reserve(program.num_constant_tensors);

  // Load each constant tensor by name
  for (uint32_t tid = 0; tid < program.num_constant_tensors; ++tid) {
    // Get tensor metadata
    if (tid >= program.tensor_meta.size() || !program.tensor_meta[tid]) {
      throw std::runtime_error(
          "load_constants: missing metadata for constant " +
          std::to_string(tid));
    }

    // Find the name for this tensor ID from named_slots
    const std::string* name = nullptr;
    for (const auto& ns : program.named_slots) {
      if (ns.slot.slot_type == SlotType::TensorSlot && ns.slot.idx == tid) {
        name = &ns.name;
        break;
      }
    }
    if (!name) {
      throw std::runtime_error(
          "load_constants: no name found for constant tensor " +
          std::to_string(tid));
    }

    // Get data from named_data_map
    if (named_data_map == nullptr) {
      throw std::runtime_error(
          "load_constants: named_data_map is null but program has constants");
    }

    auto data_result = named_data_map->get_data(name->c_str());
    if (!data_result.ok()) {
      throw std::runtime_error(
          "load_constants: failed to get data for constant '" + *name +
          "': error " + std::to_string(static_cast<int>(data_result.error())));
    }

    // Move the buffer into our storage (keeps it alive for zero-copy)
    constant_buffers.push_back(std::move(data_result.get()));
    runtime::FreeableBuffer& buffer = constant_buffers.back();

    const auto& meta = *program.tensor_meta[tid];
    Shape shape = to_shape(meta.shape);
    Dtype dtype = resolve_dtype(meta.scalar_type);

    // Create MLX array with zero-copy when enabled
    void* data_ptr = const_cast<void*>(buffer.data());

    if constexpr (kEnableConstantZeroCopy) {
      // Zero-copy: wrap pointer directly with no-op deleter
      // The FreeableBuffer in constant_buffers keeps the data alive
      auto deleter = [](void*) {
        // Data lifetime managed by FreeableBuffer in
        // MLXHandle::constant_buffers
      };
      array arr = array(data_ptr, shape, dtype, deleter);
      store.add(std::move(arr));
    } else {
      // No deleter = MLX copies the data into its own memory
      store.add(array(static_cast<const char*>(data_ptr), shape, dtype));
    }
  }

  // Evaluate all constants immediately to prepare Metal buffers
  // This trades init time for faster first inference
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

  // Pre-size the storage to fit all tensor IDs
  // Mutable buffer IDs are in the global tensor ID space
  uint32_t max_tid = 0;
  for (const auto& slot : program.mutable_buffer_map) {
    if (slot.idx > max_tid) {
      max_tid = slot.idx;
    }
  }
  store.resize(max_tid + 1);

  for (const auto& slot : program.mutable_buffer_map) {
    if (slot.slot_type != SlotType::TensorSlot) {
      throw std::runtime_error(
          "load_mutable_buffers: unexpected slot type " +
          std::to_string(static_cast<int>(slot.slot_type)));
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
    auto dtype = resolve_dtype(meta.scalar_type);

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
