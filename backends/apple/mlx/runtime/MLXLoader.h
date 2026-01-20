//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// ============================================================================
// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// ============================================================================
//
// This file was generated from schema.fbs by the MLX delegate code generator.
//
// Source:    backends/apple/mlx/serialization/schema.fbs
// Generator: backends/apple/mlx/serialization/generate.py
//
// To regenerate, run from the executorch root:
//     python backends/apple/mlx/serialization/generate.py
//
// ============================================================================
//

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "schema_generated.h"

namespace executorch {
namespace backends {
namespace mlx {

// =============================================================================
// Core types matching the Python side
// =============================================================================

struct Tid {
  uint32_t idx{};
};

template <typename T>
struct Vid {
  uint32_t idx{};
};

enum class DTypeId : int {
  f16,
  f32,
  bf16,
  i32,
  i64,
  u32,
  u8,
  boolean,
  i8,
};

// =============================================================================
// Tensor metadata
// =============================================================================

struct TensorMeta {
  std::vector<std::variant<int64_t, Vid<int32_t>>> shape;
  DTypeId dtype;
  std::vector<int32_t> strides;
};

// =============================================================================
// Constant segment info
// =============================================================================

struct ConstantSegment {
  uint64_t offset;
  uint64_t size;
};

// =============================================================================
// Op node types (AUTO-GENERATED from schema.fbs)
// =============================================================================

struct NoopNode {
};

struct LinearNode {
  Tid x;
  Tid weight;
  Tid out;
  std::optional<Tid> bias;
};

struct ItemIntNode {
  Tid x;
  Vid<int32_t> out;
};

struct ExpandDimsNode {
  Tid x;
  Tid out;
  int32_t axis;
};

struct TileNode {
  Tid x;
  Tid out;
  std::vector<int32_t> reps;
};

struct TakeAlongAxisNode {
  Tid x;
  Tid indices;
  Tid out;
  int32_t axis;
};

struct RMSNormNode {
  Tid x;
  Tid weight;
  Tid out;
  float eps;
};

struct LayerNormNode {
  Tid x;
  Tid out;
  std::optional<Tid> weight;
  std::optional<Tid> bias;
  float eps;
};

struct RopeNode {
  Tid q_in;
  Tid k_in;
  Tid q_out;
  Tid k_out;
  int32_t head_dim;
  Vid<int32_t> pos;
  std::optional<Tid> freqs;
  bool traditional;
  float base;
  float scale;
};

struct SdpaNode {
  Tid q;
  Tid k;
  Tid v;
  Tid out;
  float scale;
  std::optional<Tid> mask;
  bool causal;
};

struct AddNode {
  Tid a;
  Tid b;
  Tid out;
};

struct AddScalarNode {
  std::variant<int64_t, Vid<int32_t>> a;
  std::variant<int64_t, Vid<int32_t>> b;
  Vid<int32_t> out;
};

struct SymSizeNode {
  Tid a;
  int32_t dim;
  Vid<int32_t> out;
};

struct MulNode {
  Tid a;
  Tid b;
  Tid out;
};

struct Conv1DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
  int32_t groups;
};

struct GeluNode {
  Tid x;
  Tid out;
};

struct ARangeNode {
  Tid out;
  int32_t start;
  int32_t stop;
  int32_t step;
  DTypeId dtype;
};

struct SiluNode {
  Tid x;
  Tid out;
};

struct ReshapeNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid<int32_t>>> shape;
};

struct TransposeNode {
  Tid x;
  Tid out;
  std::vector<int32_t> perm;
};

struct ContiguousNode {
  Tid x;
  Tid out;
};

struct IdCopyNode {
  Tid x;
  Tid out;
};

struct GatherNode {
  Tid table_;
  Tid ids;
  Tid out;
};

struct SliceNode {
  Tid x;
  Tid out;
  std::variant<int64_t, Vid<int32_t>> axis;
  std::variant<int64_t, Vid<int32_t>> start;
  std::variant<int64_t, Vid<int32_t>> end;
};

struct CastNode {
  Tid x;
  Tid out;
  DTypeId dtype;
};

struct QuantizedLinearNode {
  Tid x;
  Tid w;
  Tid scales;
  Tid out;
  std::optional<Tid> biases;
  std::optional<Tid> bias;
  int32_t group_size;
  int32_t bits;
  std::string mode;
  DTypeId out_dtype;
};

struct ConcatNode {
  Tid a;
  Tid b;
  Tid out;
  int32_t axis;
};

struct FullNode {
  Tid out;
  std::vector<int32_t> shape;
  float v;
  DTypeId dtype;
};

struct ZerosNode {
  Tid out;
  std::vector<int32_t> shape;
  DTypeId dtype;
};

struct OnesNode {
  Tid out;
  std::vector<int32_t> shape;
  DTypeId dtype;
};

struct ArgmaxNode {
  Tid x;
  Tid out;
  int32_t axis;
};

struct SliceUpdateNode {
  Tid dst;
  Tid update;
  std::variant<int64_t, Vid<int32_t>> axis;
  std::variant<int64_t, Vid<int32_t>> start;
  std::variant<int64_t, Vid<int32_t>> stop;
};

struct QuantizedGatherNode {
  Tid table_q;
  Tid scales;
  Tid ids;
  Tid out;
  std::optional<Tid> biases;
  int32_t group_size;
  int32_t bits;
  std::string mode;
  DTypeId out_dtype;
};

// =============================================================================
// OpCode enum (AUTO-GENERATED from schema.fbs)
// =============================================================================

enum class OpCode : uint8_t {
  NOOP,
  LINEAR,
  ITEM_INT,
  EXPAND_DIMS,
  TILE,
  TAKE_ALONG_AXIS,
  RMS_NORM,
  LAYER_NORM,
  ROPE,
  SDPA,
  ADD,
  ADD_SCALAR,
  SYM_SIZE,
  MUL,
  CONV1D,
  GELU,
  ARANGE,
  SILU,
  RESHAPE,
  TRANSPOSE,
  CONTIGUOUS,
  ID_COPY,
  GATHER,
  SLICE,
  CAST,
  QUANTIZED_LINEAR,
  CONCAT,
  FULL,
  ZEROS,
  ONES,
  ARGMAX,
  SLICE_UPDATE,
  QUANTIZED_GATHER,
  SENTINEL
};

// =============================================================================
// NodeVariant for type-erased op storage (AUTO-GENERATED)
// =============================================================================

using NodeVariant = std::variant<
    NoopNode,
    LinearNode,
    ItemIntNode,
    ExpandDimsNode,
    TileNode,
    TakeAlongAxisNode,
    RMSNormNode,
    LayerNormNode,
    RopeNode,
    SdpaNode,
    AddNode,
    AddScalarNode,
    SymSizeNode,
    MulNode,
    Conv1DNode,
    GeluNode,
    ARangeNode,
    SiluNode,
    ReshapeNode,
    TransposeNode,
    ContiguousNode,
    IdCopyNode,
    GatherNode,
    SliceNode,
    CastNode,
    QuantizedLinearNode,
    ConcatNode,
    FullNode,
    ZerosNode,
    OnesNode,
    ArgmaxNode,
    SliceUpdateNode,
    QuantizedGatherNode>
;

// =============================================================================
// Instruction
// =============================================================================

struct Instruction {
  OpCode op{OpCode::NOOP};
  NodeVariant node;

  template <typename T>
  T& get() {
    return std::get<T>(node);
  }

  template <typename T>
  const T& get() const {
    return std::get<T>(node);
  }
};

// =============================================================================
// SlotVariant for I/O mapping
// =============================================================================

enum class SlotType : uint8_t {
  TensorSlot = 0,
  IntValueSlot = 1,
  FloatValueSlot = 2,
  BoolValueSlot = 3,
};

struct SlotVariant {
  uint32_t idx;
  SlotType slot_type;
};

// =============================================================================
// Named slot (name -> slot mapping)
// =============================================================================

struct NamedSlot {
  std::string name;
  SlotVariant slot;
};

// =============================================================================
// MLXProgram - the loaded program ready for execution
// =============================================================================

struct MLXProgram {
  std::string version;

  // Tensor/value slot counts
  uint32_t num_constant_tensors{0};
  uint32_t num_non_constant_tensors{0};
  uint32_t num_non_constant_values{0};

  // Instructions
  std::vector<Instruction> instructions;

  // I/O mappings
  std::vector<SlotVariant> input_map;
  std::vector<SlotVariant> output_map;
  std::vector<SlotVariant> mutable_buffer_map;

  // Name to slot lookup
  std::vector<NamedSlot> named_slots;

  // Tensor metadata
  std::vector<std::optional<TensorMeta>> tensor_meta;

  // Constant segment info
  ConstantSegment constant_segment;

  // Pointer to constant data (set after loading)
  const uint8_t* constant_data{nullptr};

  // Helper methods
  inline uint32_t num_tensors() const {
    return num_constant_tensors + num_non_constant_tensors;
  }

  inline uint32_t num_values() const {
    return num_non_constant_values;
  }

  inline bool is_constant_tensor(Tid id) const {
    return id.idx < num_constant_tensors;
  }

  inline size_t num_inputs() const {
    return input_map.size();
  }

  inline size_t num_outputs() const {
    return output_map.size();
  }
};

// =============================================================================
// FlatBuffer loading functions
// =============================================================================

namespace loader {

// Convert FlatBuffer DTypeId to our DTypeId
inline DTypeId convert_dtype(mlx_delegate::DTypeId fb_dtype) {
  switch (fb_dtype) {
    case mlx_delegate::DTypeId_f16:
      return DTypeId::f16;
    case mlx_delegate::DTypeId_f32:
      return DTypeId::f32;
    case mlx_delegate::DTypeId_bf16:
      return DTypeId::bf16;
    case mlx_delegate::DTypeId_i32:
      return DTypeId::i32;
    case mlx_delegate::DTypeId_i64:
      return DTypeId::i64;
    case mlx_delegate::DTypeId_u32:
      return DTypeId::u32;
    case mlx_delegate::DTypeId_u8:
      return DTypeId::u8;
    case mlx_delegate::DTypeId_boolean:
      return DTypeId::boolean;
    case mlx_delegate::DTypeId_i8:
      return DTypeId::i8;
    default:
      return DTypeId::f32;
  }
}

// Convert FlatBuffer SlotType to our SlotType
inline SlotType convert_slot_type(mlx_delegate::SlotType fb_type) {
  switch (fb_type) {
    case mlx_delegate::SlotType_TensorSlot:
      return SlotType::TensorSlot;
    case mlx_delegate::SlotType_IntValueSlot:
      return SlotType::IntValueSlot;
    case mlx_delegate::SlotType_FloatValueSlot:
      return SlotType::FloatValueSlot;
    case mlx_delegate::SlotType_BoolValueSlot:
      return SlotType::BoolValueSlot;
    default:
      return SlotType::TensorSlot;
  }
}

// Convert FlatBuffer Tid
inline Tid convert_tid(const mlx_delegate::Tid* fb_tid) {
  if (!fb_tid) {
    return Tid{0};
  }
  return Tid{fb_tid->idx()};
}

// Convert FlatBuffer Vid
inline Vid<int32_t> convert_vid(const mlx_delegate::Vid* fb_vid) {
  if (!fb_vid) {
    return Vid<int32_t>{0};
  }
  return Vid<int32_t>{fb_vid->idx()};
}

// Convert FlatBuffer IntOrVid
inline std::variant<int64_t, Vid<int32_t>> convert_int_or_vid(
    const mlx_delegate::IntOrVid* fb) {
  if (!fb) {
    return int64_t{0};
  }
  if (!fb->is_vid()) {
    return fb->literal();
  }
  const auto* vid_ptr = fb->vid();
  if (!vid_ptr) {
    return int64_t{0};
  }
  return Vid<int32_t>{vid_ptr->idx()};
}

// Convert FlatBuffer SlotVariant
inline SlotVariant convert_slot_variant(const mlx_delegate::SlotVariant* fb) {
  if (!fb) {
    return SlotVariant{0, SlotType::TensorSlot};
  }
  return SlotVariant{fb->idx(), convert_slot_type(fb->slot_type())};
}

// Load an instruction from FlatBuffer
Instruction load_instruction(const mlx_delegate::Instruction* fb_instr);

// Load the full MLXProgram from FlatBuffer data
MLXProgram load_program(const void* data, size_t size);

} // namespace loader

} // namespace mlx
} // namespace backends
} // namespace executorch