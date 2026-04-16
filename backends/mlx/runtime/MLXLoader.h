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
// Source:    backends/mlx/serialization/schema.fbs
// Generator: backends/mlx/serialization/generate.py
//
// To regenerate, run from the executorch root:
//     python backends/mlx/serialization/generate.py
//
// ============================================================================
//
// -*- c++ -*-

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "schema_generated.h"

// ExecuTorch scalar type for dtype representation
#include <executorch/runtime/core/portable_type/scalar_type.h>

namespace executorch {
namespace backends {
namespace mlx {

// =============================================================================
// Core types matching the Python side
// =============================================================================

struct Tid {
  uint32_t idx{};
};

struct Vid {
  uint32_t idx{};
};

// =============================================================================
// Tensor metadata
// =============================================================================

// Import ScalarType from ExecuTorch
using ScalarType = ::executorch::runtime::etensor::ScalarType;

struct ShapeDim {
  int32_t value{-1};       // Static dim (>= 0), or -1 for dynamic
  int32_t min_value{0};    // Lower bound (when value == -1)
  int32_t max_value{-1};   // Upper bound (-1 = unbounded, when value == -1)

  bool is_dynamic() const { return value < 0; }
};

struct TensorMeta {
  std::vector<ShapeDim> shape;
  ScalarType scalar_type{ScalarType::Float};  // ET ScalarType
  std::vector<uint8_t> dim_order;
};

// VidOrTid: either a scalar value (Vid) or a tensor (Tid)
struct VidOrTid {
  Vid vid{};
  Tid tid{};
  bool is_vid{false};  // false = use tid, true = use vid
};

// IntOrVidOrTid: a literal int, a runtime Vid, or a tensor (Tid)
struct IntOrVidOrTid {
  int64_t literal{0};
  Vid vid{};
  Tid tid{};
  uint8_t kind{0};  // 0 = literal int, 1 = vid, 2 = tid
};

// =============================================================================
// Op node types (AUTO-GENERATED from schema.fbs)
// =============================================================================

struct NoopNode {
};

struct IdCopyNode {
  Tid x;
  Tid out;
};

struct AddmmNode {
  Tid mat1;
  Tid mat2;
  Tid out;
  std::optional<Tid> bias;
  float alpha;
  float beta;
};

struct ItemIntNode {
  Tid x;
  Vid out;
};

struct ExpandDimsNode {
  Tid x;
  Tid out;
  int32_t axis;
};

struct TileNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid>> reps;
};

struct TakeAlongAxisNode {
  Tid x;
  Tid indices;
  Tid out;
  int32_t axis;
};

struct TakeNode {
  Tid x;
  Tid out;
  IntOrVidOrTid index;
  int32_t axis;
};

struct RMSNormNode {
  Tid x;
  std::optional<Tid> weight;
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
  Tid x;
  Tid out;
  int32_t dims;
  VidOrTid offset;
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

struct AddIntNode {
  std::variant<int64_t, Vid> a;
  std::variant<int64_t, Vid> b;
  Vid out;
};

struct SubtractIntNode {
  std::variant<int64_t, Vid> a;
  std::variant<int64_t, Vid> b;
  Vid out;
};

struct MultiplyIntNode {
  std::variant<int64_t, Vid> a;
  std::variant<int64_t, Vid> b;
  Vid out;
};

struct FloorDivideIntNode {
  std::variant<int64_t, Vid> a;
  std::variant<int64_t, Vid> b;
  Vid out;
};

struct ModIntNode {
  std::variant<int64_t, Vid> a;
  std::variant<int64_t, Vid> b;
  Vid out;
};

struct SymSizeNode {
  Tid a;
  int32_t dim;
  Vid out;
};

struct MultiplyNode {
  Tid a;
  Tid b;
  Tid out;
};

struct DivideNode {
  Tid a;
  Tid b;
  Tid out;
};

struct SubtractNode {
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

struct Conv2DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride_h;
  int32_t stride_w;
  int32_t padding_h;
  int32_t padding_w;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t groups;
};

struct Conv3DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride_d;
  int32_t stride_h;
  int32_t stride_w;
  int32_t padding_d;
  int32_t padding_h;
  int32_t padding_w;
  int32_t dilation_d;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t groups;
};

struct ConvTranspose1DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
  int32_t output_padding;
  int32_t groups;
};

struct ConvTranspose2DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride_h;
  int32_t stride_w;
  int32_t padding_h;
  int32_t padding_w;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t output_padding_h;
  int32_t output_padding_w;
  int32_t groups;
};

struct ConvTranspose3DNode {
  Tid x;
  Tid w;
  Tid out;
  int32_t stride_d;
  int32_t stride_h;
  int32_t stride_w;
  int32_t padding_d;
  int32_t padding_h;
  int32_t padding_w;
  int32_t dilation_d;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t output_padding_d;
  int32_t output_padding_h;
  int32_t output_padding_w;
  int32_t groups;
};

struct GeluNode {
  Tid x;
  Tid out;
  std::string approximate;
};

struct ARangeNode {
  Tid out;
  std::variant<int64_t, Vid> start;
  std::variant<int64_t, Vid> stop;
  std::variant<int64_t, Vid> step;
  std::optional<int8_t> scalar_type;
};

struct SiluNode {
  Tid x;
  Tid out;
};

struct SigmoidNode {
  Tid x;
  Tid out;
};

struct TanhNode {
  Tid x;
  Tid out;
};

struct SqueezeNode {
  Tid x;
  Tid out;
  std::vector<int32_t> dims;
};

struct SplitNode {
  Tid x;
  std::vector<Tid> outs;
  std::vector<std::variant<int64_t, Vid>> sizes;
  int32_t axis;
};

struct RsqrtNode {
  Tid x;
  Tid out;
};

struct MaximumNode {
  Tid a;
  Tid b;
  Tid out;
};

struct MinimumNode {
  Tid a;
  Tid b;
  Tid out;
};

struct LogNode {
  Tid x;
  Tid out;
};

struct SoftmaxNode {
  Tid x;
  Tid out;
  int32_t axis;
  bool precise;
};

struct BroadcastToNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid>> shape;
};

struct PadNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid>> pad_width;
  std::string mode;
  float constant_value;
};

struct WhereNode {
  Tid condition;
  Tid x;
  Tid y;
  Tid out;
};

struct ReshapeNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid>> shape;
};

struct TransposeNode {
  Tid x;
  Tid out;
  std::vector<int32_t> perm;
};

struct AsStridedNode {
  Tid x;
  Tid out;
  std::vector<std::variant<int64_t, Vid>> shape;
  std::vector<std::variant<int64_t, Vid>> strides;
  uint64_t offset;
};

struct ContiguousNode {
  Tid x;
  Tid out;
};

struct GatherNode {
  Tid x;
  std::vector<Tid> indices;
  Tid out;
  std::vector<int32_t> axes;
  std::vector<int32_t> slice_sizes;
};

struct SliceNode {
  Tid x;
  Tid out;
  std::variant<int64_t, Vid> axis;
  std::variant<int64_t, Vid> start;
  std::variant<int64_t, Vid> stop;
  int32_t step;
};

struct AsTypeNode {
  Tid x;
  Tid out;
  int8_t scalar_type;
};

struct QuantizedMatmulNode {
  Tid x;
  Tid w;
  Tid scales;
  Tid out;
  std::optional<Tid> biases;
  int32_t group_size;
  int32_t bits;
  std::string mode;
  bool transpose;
};

struct ScatterAddNode {
  Tid x;
  Tid indices;
  Tid updates;
  Tid out;
  int32_t axis;
};

struct ConcatenateNode {
  std::vector<Tid> tensors;
  Tid out;
  int32_t axis;
};

struct FullNode {
  Tid out;
  std::vector<std::variant<int64_t, Vid>> shape;
  std::variant<double, Vid> v;
  int8_t scalar_type;
};

struct FullLikeNode {
  Tid x;
  Tid out;
  std::variant<double, Vid> v;
  std::optional<int8_t> scalar_type;
};

struct ArgmaxNode {
  Tid x;
  Tid out;
  int32_t axis;
  bool keepdims;
};

struct SliceUpdateNode {
  Tid dst;
  Tid update;
  Tid out;
  std::variant<int64_t, Vid> axis;
  std::variant<int64_t, Vid> start;
  std::variant<int64_t, Vid> stop;
  int32_t step;
};

struct IndexCopyNode {
  Tid dst;
  Tid update;
  Tid indices;
  Tid out;
  int32_t axis;
};

struct DequantizeNode {
  Tid w;
  Tid scales;
  Tid out;
  std::optional<Tid> biases;
  int32_t group_size;
  int32_t bits;
  std::string mode;
  std::optional<Tid> global_scale;
  std::optional<int8_t> dtype;
};

struct LessNode {
  Tid a;
  Tid b;
  Tid out;
};

struct LessEqualNode {
  Tid a;
  Tid b;
  Tid out;
};

struct GreaterNode {
  Tid a;
  Tid b;
  Tid out;
};

struct GreaterEqualNode {
  Tid a;
  Tid b;
  Tid out;
};

struct EqualNode {
  Tid a;
  Tid b;
  Tid out;
};

struct NotEqualNode {
  Tid a;
  Tid b;
  Tid out;
};

struct LogicalNotNode {
  Tid x;
  Tid out;
};

struct LogicalAndNode {
  Tid a;
  Tid b;
  Tid out;
};

struct LogicalOrNode {
  Tid a;
  Tid b;
  Tid out;
};

struct TriNode {
  Tid out;
  std::variant<int64_t, Vid> n;
  std::variant<int64_t, Vid> m;
  int32_t k;
  int8_t scalar_type;
};

struct TrilNode {
  Tid x;
  Tid out;
  int32_t k;
};

struct TriuNode {
  Tid x;
  Tid out;
  int32_t k;
};

struct ClipNode {
  Tid x;
  Tid out;
  std::optional<Tid> a_min;
  std::optional<Tid> a_max;
};

struct CumsumNode {
  Tid x;
  Tid out;
  int32_t axis;
  bool reverse;
  bool inclusive;
};

struct StackNode {
  std::vector<Tid> tensors;
  Tid out;
  int32_t axis;
};

struct SignNode {
  Tid x;
  Tid out;
};

struct AnyNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct AllNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct RepeatNode {
  Tid x;
  Tid out;
  std::variant<int64_t, Vid> repeats;
  int32_t axis;
};

struct SortNode {
  Tid x;
  Tid out;
  int32_t axis;
};

struct ArgsortNode {
  Tid x;
  Tid out;
  int32_t axis;
};

struct PartitionNode {
  Tid x;
  Tid out;
  std::variant<int64_t, Vid> kth;
  int32_t axis;
};

struct ArgPartitionNode {
  Tid x;
  Tid out;
  std::variant<int64_t, Vid> kth;
  int32_t axis;
};

struct FloorNode {
  Tid x;
  Tid out;
};

struct CeilNode {
  Tid x;
  Tid out;
};

struct SquareNode {
  Tid x;
  Tid out;
};

struct ExpNode {
  Tid x;
  Tid out;
};

struct SinNode {
  Tid x;
  Tid out;
};

struct CosNode {
  Tid x;
  Tid out;
};

struct TanNode {
  Tid x;
  Tid out;
};

struct ArcsinNode {
  Tid x;
  Tid out;
};

struct ArccosNode {
  Tid x;
  Tid out;
};

struct ArctanNode {
  Tid x;
  Tid out;
};

struct SinhNode {
  Tid x;
  Tid out;
};

struct CoshNode {
  Tid x;
  Tid out;
};

struct ArcsinhNode {
  Tid x;
  Tid out;
};

struct ArccoshNode {
  Tid x;
  Tid out;
};

struct ArctanhNode {
  Tid x;
  Tid out;
};

struct Log2Node {
  Tid x;
  Tid out;
};

struct Log10Node {
  Tid x;
  Tid out;
};

struct Log1pNode {
  Tid x;
  Tid out;
};

struct ErfNode {
  Tid x;
  Tid out;
};

struct Expm1Node {
  Tid x;
  Tid out;
};

struct RoundNode {
  Tid x;
  Tid out;
  int32_t decimals;
};

struct ReciprocalNode {
  Tid x;
  Tid out;
};

struct SqrtNode {
  Tid x;
  Tid out;
};

struct AbsNode {
  Tid x;
  Tid out;
};

struct NegNode {
  Tid x;
  Tid out;
};

struct Atan2Node {
  Tid a;
  Tid b;
  Tid out;
};

struct LogAddExpNode {
  Tid a;
  Tid b;
  Tid out;
};

struct FloorDivideNode {
  Tid a;
  Tid b;
  Tid out;
};

struct RemainderNode {
  Tid a;
  Tid b;
  Tid out;
};

struct PowerNode {
  Tid a;
  Tid b;
  Tid out;
};

struct LogSumExpNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct SumNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct MeanNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct VarNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
  int32_t ddof;
};

struct StdNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
  int32_t ddof;
};

struct ProdNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct MaxNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct MinNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct ArgminNode {
  Tid x;
  Tid out;
  int32_t axis;
  bool keepdims;
};

struct MedianNode {
  Tid x;
  Tid out;
  std::vector<int32_t> axes;
  bool keepdims;
};

struct GatherMmNode {
  Tid a;
  Tid b;
  Tid out;
  std::optional<Tid> lhs_indices;
  std::optional<Tid> rhs_indices;
  bool sorted_indices;
};

struct GatherQmmNode {
  Tid x;
  Tid w;
  Tid scales;
  Tid out;
  std::string mode;
  std::optional<Tid> biases;
  std::optional<Tid> lhs_indices;
  std::optional<Tid> rhs_indices;
  bool transpose;
  int32_t group_size;
  int32_t bits;
  bool sorted_indices;
};

struct ScanNode {
  std::vector<Tid> originals;
  std::vector<Tid> sliced;
  std::vector<Tid> outputs;
  std::vector<Tid> carry;
  int32_t body_chain_idx;
  int32_t scan_axis;
};

struct MetalKernelNode {
  std::string name;
  std::string source;
  std::vector<Tid> inputs;
  std::vector<Tid> outputs;
  std::vector<std::variant<int64_t, Vid>> grid;
  std::vector<std::variant<int64_t, Vid>> threadgroup;
  std::string header;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool ensure_row_contiguous;
  bool atomic_outputs;
  std::vector<std::variant<int64_t, Vid>> output_shapes_flat;
  std::vector<int32_t> output_shape_lengths;
  std::vector<int8_t> output_dtypes;
  std::vector<std::string> template_arg_names;
  std::vector<int8_t> template_arg_kinds;
  std::vector<int32_t> template_arg_values;
  std::optional<float> init_value;
};

struct BitwiseXorNode {
  std::optional<Tid> a;
  std::optional<Tid> b;
  std::optional<Tid> out;
};


// =============================================================================
// OpCode enum (AUTO-GENERATED from schema.fbs)
// =============================================================================

enum class OpCode : uint8_t {
  NOOP,
  ID_COPY,
  ADDMM,
  ITEM_INT,
  EXPAND_DIMS,
  TILE,
  TAKE_ALONG_AXIS,
  TAKE,
  RMS_NORM,
  LAYER_NORM,
  ROPE,
  SDPA,
  ADD,
  ADD_INT,
  SUBTRACT_INT,
  MULTIPLY_INT,
  FLOOR_DIVIDE_INT,
  MOD_INT,
  SYM_SIZE,
  MULTIPLY,
  DIVIDE,
  SUBTRACT,
  CONV1D,
  CONV2D,
  CONV3D,
  CONV_TRANSPOSE1D,
  CONV_TRANSPOSE2D,
  CONV_TRANSPOSE3D,
  GELU,
  ARANGE,
  SILU,
  SIGMOID,
  TANH,
  SQUEEZE,
  SPLIT,
  RSQRT,
  MAXIMUM,
  MINIMUM,
  LOG,
  SOFTMAX,
  BROADCAST_TO,
  PAD,
  WHERE,
  RESHAPE,
  TRANSPOSE,
  AS_STRIDED,
  CONTIGUOUS,
  GATHER,
  SLICE,
  ASTYPE,
  QUANTIZED_MATMUL,
  SCATTER_ADD,
  CONCATENATE,
  FULL,
  FULL_LIKE,
  ARGMAX,
  SLICE_UPDATE,
  INDEX_COPY,
  DEQUANTIZE,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
  EQUAL,
  NOT_EQUAL,
  LOGICAL_NOT,
  LOGICAL_AND,
  LOGICAL_OR,
  TRI,
  TRIL,
  TRIU,
  CLIP,
  CUMSUM,
  STACK,
  SIGN,
  ANY,
  ALL,
  REPEAT,
  SORT,
  ARGSORT,
  PARTITION,
  ARG_PARTITION,
  FLOOR,
  CEIL,
  SQUARE,
  EXP,
  SIN,
  COS,
  TAN,
  ARCSIN,
  ARCCOS,
  ARCTAN,
  SINH,
  COSH,
  ARCSINH,
  ARCCOSH,
  ARCTANH,
  LOG2,
  LOG10,
  LOG1P,
  ERF,
  EXPM1,
  ROUND,
  RECIPROCAL,
  SQRT,
  ABS,
  NEG,
  ATAN2,
  LOG_ADD_EXP,
  FLOOR_DIVIDE,
  REMAINDER,
  POWER,
  LOG_SUM_EXP,
  SUM,
  MEAN,
  VAR,
  STD,
  PROD,
  MAX,
  MIN,
  ARGMIN,
  MEDIAN,
  GATHER_MM,
  GATHER_QMM,
  SCAN,
  METAL_KERNEL,
  BITWISE_XOR,
};

// OpCode to string conversion (for logging)
inline const char* op_name(OpCode op) {
  switch (op) {
    case OpCode::NOOP:
      return "NOOP";
    case OpCode::ID_COPY:
      return "ID_COPY";
    case OpCode::ADDMM:
      return "ADDMM";
    case OpCode::ITEM_INT:
      return "ITEM_INT";
    case OpCode::EXPAND_DIMS:
      return "EXPAND_DIMS";
    case OpCode::TILE:
      return "TILE";
    case OpCode::TAKE_ALONG_AXIS:
      return "TAKE_ALONG_AXIS";
    case OpCode::TAKE:
      return "TAKE";
    case OpCode::RMS_NORM:
      return "RMS_NORM";
    case OpCode::LAYER_NORM:
      return "LAYER_NORM";
    case OpCode::ROPE:
      return "ROPE";
    case OpCode::SDPA:
      return "SDPA";
    case OpCode::ADD:
      return "ADD";
    case OpCode::ADD_INT:
      return "ADD_INT";
    case OpCode::SUBTRACT_INT:
      return "SUBTRACT_INT";
    case OpCode::MULTIPLY_INT:
      return "MULTIPLY_INT";
    case OpCode::FLOOR_DIVIDE_INT:
      return "FLOOR_DIVIDE_INT";
    case OpCode::MOD_INT:
      return "MOD_INT";
    case OpCode::SYM_SIZE:
      return "SYM_SIZE";
    case OpCode::MULTIPLY:
      return "MULTIPLY";
    case OpCode::DIVIDE:
      return "DIVIDE";
    case OpCode::SUBTRACT:
      return "SUBTRACT";
    case OpCode::CONV1D:
      return "CONV1D";
    case OpCode::CONV2D:
      return "CONV2D";
    case OpCode::CONV3D:
      return "CONV3D";
    case OpCode::CONV_TRANSPOSE1D:
      return "CONV_TRANSPOSE1D";
    case OpCode::CONV_TRANSPOSE2D:
      return "CONV_TRANSPOSE2D";
    case OpCode::CONV_TRANSPOSE3D:
      return "CONV_TRANSPOSE3D";
    case OpCode::GELU:
      return "GELU";
    case OpCode::ARANGE:
      return "ARANGE";
    case OpCode::SILU:
      return "SILU";
    case OpCode::SIGMOID:
      return "SIGMOID";
    case OpCode::TANH:
      return "TANH";
    case OpCode::SQUEEZE:
      return "SQUEEZE";
    case OpCode::SPLIT:
      return "SPLIT";
    case OpCode::RSQRT:
      return "RSQRT";
    case OpCode::MAXIMUM:
      return "MAXIMUM";
    case OpCode::MINIMUM:
      return "MINIMUM";
    case OpCode::LOG:
      return "LOG";
    case OpCode::SOFTMAX:
      return "SOFTMAX";
    case OpCode::BROADCAST_TO:
      return "BROADCAST_TO";
    case OpCode::PAD:
      return "PAD";
    case OpCode::WHERE:
      return "WHERE";
    case OpCode::RESHAPE:
      return "RESHAPE";
    case OpCode::TRANSPOSE:
      return "TRANSPOSE";
    case OpCode::AS_STRIDED:
      return "AS_STRIDED";
    case OpCode::CONTIGUOUS:
      return "CONTIGUOUS";
    case OpCode::GATHER:
      return "GATHER";
    case OpCode::SLICE:
      return "SLICE";
    case OpCode::ASTYPE:
      return "ASTYPE";
    case OpCode::QUANTIZED_MATMUL:
      return "QUANTIZED_MATMUL";
    case OpCode::SCATTER_ADD:
      return "SCATTER_ADD";
    case OpCode::CONCATENATE:
      return "CONCATENATE";
    case OpCode::FULL:
      return "FULL";
    case OpCode::FULL_LIKE:
      return "FULL_LIKE";
    case OpCode::ARGMAX:
      return "ARGMAX";
    case OpCode::SLICE_UPDATE:
      return "SLICE_UPDATE";
    case OpCode::INDEX_COPY:
      return "INDEX_COPY";
    case OpCode::DEQUANTIZE:
      return "DEQUANTIZE";
    case OpCode::LESS:
      return "LESS";
    case OpCode::LESS_EQUAL:
      return "LESS_EQUAL";
    case OpCode::GREATER:
      return "GREATER";
    case OpCode::GREATER_EQUAL:
      return "GREATER_EQUAL";
    case OpCode::EQUAL:
      return "EQUAL";
    case OpCode::NOT_EQUAL:
      return "NOT_EQUAL";
    case OpCode::LOGICAL_NOT:
      return "LOGICAL_NOT";
    case OpCode::LOGICAL_AND:
      return "LOGICAL_AND";
    case OpCode::LOGICAL_OR:
      return "LOGICAL_OR";
    case OpCode::TRI:
      return "TRI";
    case OpCode::TRIL:
      return "TRIL";
    case OpCode::TRIU:
      return "TRIU";
    case OpCode::CLIP:
      return "CLIP";
    case OpCode::CUMSUM:
      return "CUMSUM";
    case OpCode::STACK:
      return "STACK";
    case OpCode::SIGN:
      return "SIGN";
    case OpCode::ANY:
      return "ANY";
    case OpCode::ALL:
      return "ALL";
    case OpCode::REPEAT:
      return "REPEAT";
    case OpCode::SORT:
      return "SORT";
    case OpCode::ARGSORT:
      return "ARGSORT";
    case OpCode::PARTITION:
      return "PARTITION";
    case OpCode::ARG_PARTITION:
      return "ARG_PARTITION";
    case OpCode::FLOOR:
      return "FLOOR";
    case OpCode::CEIL:
      return "CEIL";
    case OpCode::SQUARE:
      return "SQUARE";
    case OpCode::EXP:
      return "EXP";
    case OpCode::SIN:
      return "SIN";
    case OpCode::COS:
      return "COS";
    case OpCode::TAN:
      return "TAN";
    case OpCode::ARCSIN:
      return "ARCSIN";
    case OpCode::ARCCOS:
      return "ARCCOS";
    case OpCode::ARCTAN:
      return "ARCTAN";
    case OpCode::SINH:
      return "SINH";
    case OpCode::COSH:
      return "COSH";
    case OpCode::ARCSINH:
      return "ARCSINH";
    case OpCode::ARCCOSH:
      return "ARCCOSH";
    case OpCode::ARCTANH:
      return "ARCTANH";
    case OpCode::LOG2:
      return "LOG2";
    case OpCode::LOG10:
      return "LOG10";
    case OpCode::LOG1P:
      return "LOG1P";
    case OpCode::ERF:
      return "ERF";
    case OpCode::EXPM1:
      return "EXPM1";
    case OpCode::ROUND:
      return "ROUND";
    case OpCode::RECIPROCAL:
      return "RECIPROCAL";
    case OpCode::SQRT:
      return "SQRT";
    case OpCode::ABS:
      return "ABS";
    case OpCode::NEG:
      return "NEG";
    case OpCode::ATAN2:
      return "ATAN2";
    case OpCode::LOG_ADD_EXP:
      return "LOG_ADD_EXP";
    case OpCode::FLOOR_DIVIDE:
      return "FLOOR_DIVIDE";
    case OpCode::REMAINDER:
      return "REMAINDER";
    case OpCode::POWER:
      return "POWER";
    case OpCode::LOG_SUM_EXP:
      return "LOG_SUM_EXP";
    case OpCode::SUM:
      return "SUM";
    case OpCode::MEAN:
      return "MEAN";
    case OpCode::VAR:
      return "VAR";
    case OpCode::STD:
      return "STD";
    case OpCode::PROD:
      return "PROD";
    case OpCode::MAX:
      return "MAX";
    case OpCode::MIN:
      return "MIN";
    case OpCode::ARGMIN:
      return "ARGMIN";
    case OpCode::MEDIAN:
      return "MEDIAN";
    case OpCode::GATHER_MM:
      return "GATHER_MM";
    case OpCode::GATHER_QMM:
      return "GATHER_QMM";
    case OpCode::SCAN:
      return "SCAN";
    case OpCode::METAL_KERNEL:
      return "METAL_KERNEL";
    case OpCode::BITWISE_XOR:
      return "BITWISE_XOR";
  }
  return "UNKNOWN";
}

// =============================================================================
// NodeVariant for type-erased op storage (AUTO-GENERATED)
// =============================================================================

using NodeVariant = std::variant<
    NoopNode,
    IdCopyNode,
    AddmmNode,
    ItemIntNode,
    ExpandDimsNode,
    TileNode,
    TakeAlongAxisNode,
    TakeNode,
    RMSNormNode,
    LayerNormNode,
    RopeNode,
    SdpaNode,
    AddNode,
    AddIntNode,
    SubtractIntNode,
    MultiplyIntNode,
    FloorDivideIntNode,
    ModIntNode,
    SymSizeNode,
    MultiplyNode,
    DivideNode,
    SubtractNode,
    Conv1DNode,
    Conv2DNode,
    Conv3DNode,
    ConvTranspose1DNode,
    ConvTranspose2DNode,
    ConvTranspose3DNode,
    GeluNode,
    ARangeNode,
    SiluNode,
    SigmoidNode,
    TanhNode,
    SqueezeNode,
    SplitNode,
    RsqrtNode,
    MaximumNode,
    MinimumNode,
    LogNode,
    SoftmaxNode,
    BroadcastToNode,
    PadNode,
    WhereNode,
    ReshapeNode,
    TransposeNode,
    AsStridedNode,
    ContiguousNode,
    GatherNode,
    SliceNode,
    AsTypeNode,
    QuantizedMatmulNode,
    ScatterAddNode,
    ConcatenateNode,
    FullNode,
    FullLikeNode,
    ArgmaxNode,
    SliceUpdateNode,
    IndexCopyNode,
    DequantizeNode,
    LessNode,
    LessEqualNode,
    GreaterNode,
    GreaterEqualNode,
    EqualNode,
    NotEqualNode,
    LogicalNotNode,
    LogicalAndNode,
    LogicalOrNode,
    TriNode,
    TrilNode,
    TriuNode,
    ClipNode,
    CumsumNode,
    StackNode,
    SignNode,
    AnyNode,
    AllNode,
    RepeatNode,
    SortNode,
    ArgsortNode,
    PartitionNode,
    ArgPartitionNode,
    FloorNode,
    CeilNode,
    SquareNode,
    ExpNode,
    SinNode,
    CosNode,
    TanNode,
    ArcsinNode,
    ArccosNode,
    ArctanNode,
    SinhNode,
    CoshNode,
    ArcsinhNode,
    ArccoshNode,
    ArctanhNode,
    Log2Node,
    Log10Node,
    Log1pNode,
    ErfNode,
    Expm1Node,
    RoundNode,
    ReciprocalNode,
    SqrtNode,
    AbsNode,
    NegNode,
    Atan2Node,
    LogAddExpNode,
    FloorDivideNode,
    RemainderNode,
    PowerNode,
    LogSumExpNode,
    SumNode,
    MeanNode,
    VarNode,
    StdNode,
    ProdNode,
    MaxNode,
    MinNode,
    ArgminNode,
    MedianNode,
    GatherMmNode,
    GatherQmmNode,
    ScanNode,
    MetalKernelNode,
    BitwiseXorNode
>;

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

  // Tensor/value slot counts (in Tid assignment order)
  uint32_t num_constant_tensors{0};
  uint32_t num_input_tensors{0};
  uint32_t num_output_tensors{0};
  uint32_t num_mutable_buffer_tensors{0};
  uint32_t num_temp_tensors{0};
  uint32_t num_values{0};

  // Instruction chains
  std::vector<std::vector<Instruction>> instruction_chains;
  uint32_t main_chain_idx{0};
  int32_t init_chain_idx{-1};  // -1 = no init chain

  // I/O mappings
  std::vector<SlotVariant> input_map;
  std::vector<SlotVariant> output_map;
  std::vector<SlotVariant> mutable_buffer_map;

  // Name to slot lookup
  std::vector<NamedSlot> named_slots;

  // Tensor metadata
  std::vector<std::optional<TensorMeta>> tensor_meta;

  // Helper methods
  inline uint64_t num_tensors() const {
    return static_cast<uint64_t>(num_constant_tensors) +
           num_input_tensors + num_output_tensors +
           num_mutable_buffer_tensors + num_temp_tensors;
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
      throw std::runtime_error("Unknown SlotType: " +
          std::to_string(static_cast<int>(fb_type)));
  }
}

// Convert FlatBuffer Tid
inline Tid convert_tid(const mlx_delegate::Tid* fb_tid) {
  if (!fb_tid) {
    throw std::runtime_error("Null Tid in FlatBuffer");
  }
  return Tid{fb_tid->idx()};
}

// Convert FlatBuffer Vid
inline Vid convert_vid(const mlx_delegate::Vid* fb_vid) {
  if (!fb_vid) {
    throw std::runtime_error("Null Vid in FlatBuffer");
  }
  return Vid{fb_vid->idx()};
}

// Convert FlatBuffer IntOrVid
inline std::variant<int64_t, Vid> convert_int_or_vid(
    const mlx_delegate::IntOrVid* fb) {
  if (!fb) {
    throw std::runtime_error("Null IntOrVid in FlatBuffer");
  }
  if (!fb->is_vid()) {
    return fb->literal();
  }
  const auto* vid_ptr = fb->vid();
  if (!vid_ptr) {
    throw std::runtime_error("IntOrVid has is_vid=true but vid pointer is null");
  }
  return Vid{vid_ptr->idx()};
}

// Convert FlatBuffer FloatOrVid
inline std::variant<double, Vid> convert_float_or_vid(
    const mlx_delegate::FloatOrVid* fb) {
  if (!fb) {
    throw std::runtime_error("Null FloatOrVid in FlatBuffer");
  }
  if (!fb->is_vid()) {
    return fb->literal();
  }
  const auto* vid_ptr = fb->vid();
  if (!vid_ptr) {
    throw std::runtime_error("FloatOrVid has is_vid=true but vid pointer is null");
  }
  return Vid{vid_ptr->idx()};
}

// Convert FlatBuffer VidOrTid (scalar value or tensor)
inline VidOrTid convert_vid_or_tid(
    const mlx_delegate::VidOrTid* fb) {
  if (!fb) {
    throw std::runtime_error("Null VidOrTid in FlatBuffer");
  }
  VidOrTid result;
  result.is_vid = fb->is_vid();
  if (result.is_vid) {
    if (!fb->vid()) {
      throw std::runtime_error("VidOrTid has is_vid=true but vid pointer is null");
    }
    result.vid = Vid{fb->vid()->idx()};
  } else {
    if (!fb->tid()) {
      throw std::runtime_error("VidOrTid has is_vid=false but tid pointer is null");
    }
    result.tid = Tid{fb->tid()->idx()};
  }
  return result;
}

// Convert FlatBuffer IntOrVidOrTid (literal int, Vid, or Tid)
inline IntOrVidOrTid convert_int_or_vid_or_tid(
    const mlx_delegate::IntOrVidOrTid* fb) {
  if (!fb) {
    throw std::runtime_error("Null IntOrVidOrTid in FlatBuffer");
  }
  IntOrVidOrTid result;
  result.kind = fb->kind();
  switch (result.kind) {
    case 0:  // literal int
      result.literal = fb->literal();
      break;
    case 1: {  // Vid
      const auto* vid_ptr = fb->vid();
      if (!vid_ptr) {
        throw std::runtime_error(
            "IntOrVidOrTid has kind=1 (Vid) but vid pointer is null");
      }
      result.vid = Vid{vid_ptr->idx()};
      break;
    }
    case 2: {  // Tid
      const auto* tid_ptr = fb->tid();
      if (!tid_ptr) {
        throw std::runtime_error(
            "IntOrVidOrTid has kind=2 (Tid) but tid pointer is null");
      }
      result.tid = Tid{tid_ptr->idx()};
      break;
    }
    default:
      throw std::runtime_error(
          "IntOrVidOrTid has invalid kind: " + std::to_string(result.kind));
  }
  return result;
}

// Convert FlatBuffer SlotVariant
inline SlotVariant convert_slot_variant(const mlx_delegate::SlotVariant* fb) {
  if (!fb) {
    throw std::runtime_error("Null SlotVariant in FlatBuffer");
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
