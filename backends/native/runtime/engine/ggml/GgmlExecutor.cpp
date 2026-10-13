// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlExecutor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn {
namespace {

constexpr std::string_view kTorchOpsPrefix = "torch.ops.";
constexpr std::string_view kQ4KCodec = "gguf:q4_k";
constexpr std::string_view kQ6KCodec = "gguf:q6_k";
constexpr int64_t kKQuantBlockElements = 256;
constexpr size_t kQ4KBlockBytes = 144;
constexpr size_t kQ6KBlockBytes = 210;
constexpr size_t kQ40BlockElements = 32;

struct BlockQ40 {
  ggml_fp16_t d;
  uint8_t qs[kQ40BlockElements / 2];
};

size_t checked_numel(const std::vector<int64_t>& sizes) {
  size_t out = 1;
  for (const int64_t size : sizes) {
    if (size < 0 || static_cast<uint64_t>(size) > SIZE_MAX ||
        (size != 0 && out > SIZE_MAX / static_cast<size_t>(size))) {
      throw std::runtime_error("ggml: tensor shape is not representable");
    }
    out *= static_cast<size_t>(size);
  }
  return out;
}

size_t checked_nbytes(const std::vector<int64_t>& sizes, ScalarType dtype) {
  const size_t numel = checked_numel(sizes);
  const size_t width = element_size(dtype);
  if (width != 0 && numel > SIZE_MAX / width) {
    throw std::runtime_error("ggml: tensor byte size overflows");
  }
  return numel * width;
}

const PackedQuant* packed_quant(const TensorMeta& meta) {
  if (!meta.quant) {
    return nullptr;
  }
  return std::get_if<PackedQuant>(&*meta.quant);
}

size_t packed_nbytes(const TensorMeta& meta) {
  const PackedQuant* packed = packed_quant(meta);
  if (packed == nullptr ||
      (packed->codec != kQ4KCodec && packed->codec != kQ6KCodec)) {
    throw std::runtime_error(
        "ggml: only PackedQuant codecs 'gguf:q4_k' and 'gguf:q6_k' "
        "are supported");
  }
  if (meta.sizes.size() != 2 || meta.sizes[0] <= 0 || meta.sizes[1] <= 0 ||
      meta.sizes[1] % kKQuantBlockElements != 0) {
    throw std::runtime_error(
        "ggml: K-quant weight must have logical shape [O, I] with I divisible "
        "by 256");
  }
  const ggml_type type =
      packed->codec == kQ4KCodec ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
  const size_t block_bytes =
      packed->codec == kQ4KCodec ? kQ4KBlockBytes : kQ6KBlockBytes;
  if (ggml_type_size(type) != block_bytes ||
      ggml_blck_size(type) != kKQuantBlockElements) {
    throw std::runtime_error("ggml: unexpected GGUF K-quant block layout");
  }
  const size_t rows = static_cast<size_t>(meta.sizes[0]);
  const size_t blocks =
      static_cast<size_t>(meta.sizes[1] / kKQuantBlockElements);
  if (blocks > SIZE_MAX / block_bytes ||
      rows > SIZE_MAX / (blocks * block_bytes)) {
    throw std::runtime_error("ggml: K-quant byte size overflows");
  }
  return rows * blocks * block_bytes;
}

const AffineGroupQuant* affine_quant(const TensorMeta& meta) {
  if (!meta.quant) {
    return nullptr;
  }
  return std::get_if<AffineGroupQuant>(&*meta.quant);
}

// Symmetric int4 groups that are whole multiples of 32 map onto Q4_0 blocks.
// Q4_0 stores the scale as fp16, so scales are narrowed at load.
const AffineGroupQuant& q4_affine(const TensorMeta& meta) {
  const AffineGroupQuant* quant = affine_quant(meta);
  if (quant == nullptr || quant->quant_min != -8 || quant->quant_max != 7 ||
      quant->group_size <= 0 || quant->group_size % kQ40BlockElements != 0 ||
      meta.sizes.size() != 2 || meta.sizes[0] <= 0 ||
      meta.sizes[1] % quant->group_size != 0) {
    throw std::runtime_error(
        "ggml: affine weights must be int4 [N, K] with groups divisible by 32");
  }
  return *quant;
}

std::vector<float> to_float_vector(ByteSpan bytes, ScalarType dtype) {
  const size_t count = bytes.size() / std::max<size_t>(element_size(dtype), 1);
  std::vector<float> out(count);
  switch (dtype) {
    case ScalarType::Float:
      std::memcpy(out.data(), bytes.data(), count * sizeof(float));
      return out;
    case ScalarType::Half:
      ggml_fp16_to_fp32_row(
          reinterpret_cast<const ggml_fp16_t*>(bytes.data()),
          out.data(),
          static_cast<int64_t>(count));
      return out;
    case ScalarType::BFloat16:
      ggml_bf16_to_fp32_row(
          reinterpret_cast<const ggml_bf16_t*>(bytes.data()),
          out.data(),
          static_cast<int64_t>(count));
      return out;
    default:
      throw std::runtime_error(
          "ggml: expected a floating-point constant, got " +
          std::string(scalar_type_name(dtype)));
  }
}

std::vector<uint8_t> affine_int4_to_q4_0(
    const TensorMeta& meta,
    ByteSpan packed,
    const std::vector<float>& scales) {
  static_assert(
      sizeof(BlockQ40) == sizeof(ggml_fp16_t) + kQ40BlockElements / 2);
  if (ggml_type_size(GGML_TYPE_Q4_0) != sizeof(BlockQ40) ||
      ggml_blck_size(GGML_TYPE_Q4_0) != kQ40BlockElements) {
    throw std::runtime_error("ggml: unexpected Q4_0 block layout");
  }
  const AffineGroupQuant& quant = q4_affine(meta);
  const size_t rows = static_cast<size_t>(meta.sizes[0]);
  const size_t cols = static_cast<size_t>(meta.sizes[1]);
  const size_t groups = cols / static_cast<size_t>(quant.group_size);
  if (packed.size() != rows * cols / 2 || scales.size() != rows * groups) {
    throw std::runtime_error("ggml: int4 weight or scale size mismatch");
  }
  std::vector<uint8_t> out(
      rows * (cols / kQ40BlockElements) * sizeof(BlockQ40));
  auto* block = reinterpret_cast<BlockQ40*>(out.data());
  // Column c sits in the low nibble when c is even, stored as q + 8: the same
  // offset encoding Q4_0 uses, so only the nibble order changes.
  for (size_t r = 0; r < rows; ++r) {
    const uint8_t* row = packed.data() + r * cols / 2;
    const auto nibble = [row](size_t c) -> uint8_t {
      return (row[c / 2] >> (4 * (c & 1))) & 0xF;
    };
    for (size_t c0 = 0; c0 < cols; c0 += kQ40BlockElements, ++block) {
      block->d = ggml_fp32_to_fp16(
          scales[r * groups + c0 / static_cast<size_t>(quant.group_size)]);
      for (size_t j = 0; j < kQ40BlockElements / 2; ++j) {
        block->qs[j] = static_cast<uint8_t>(
            nibble(c0 + j) | (nibble(c0 + j + kQ40BlockElements / 2) << 4));
      }
    }
  }
  return out;
}

ggml_type to_ggml_type(const TensorMeta& meta) {
  if (affine_quant(meta) != nullptr) {
    q4_affine(meta);
    return GGML_TYPE_Q4_0;
  }
  if (packed_quant(meta) != nullptr) {
    packed_nbytes(meta);
    if (packed_quant(meta)->codec == kQ4KCodec) {
      return GGML_TYPE_Q4_K;
    }
    if (packed_quant(meta)->codec == kQ6KCodec) {
      return GGML_TYPE_Q6_K;
    }
    throw std::runtime_error(
        "ggml: unsupported PackedQuant codec '" + packed_quant(meta)->codec +
        "'");
  }
  switch (meta.dtype) {
    case ScalarType::Float:
      return GGML_TYPE_F32;
    case ScalarType::Half:
      return GGML_TYPE_F16;
    case ScalarType::BFloat16:
      return GGML_TYPE_BF16;
    case ScalarType::Int:
      return GGML_TYPE_I32;
    case ScalarType::Long:
      return GGML_TYPE_I64;
    // Byte, Char, Short and Double would be I8, I16 and F64, which ggml
    // cannot convert to any other type.
    default:
      throw std::runtime_error(
          "ggml: unsupported tensor dtype " +
          std::string(scalar_type_name(meta.dtype)));
  }
}

// ggml has no int64 kernels, so int64 values are carried as I32.
ggml_type compute_type(const TensorMeta& meta) {
  return meta.dtype == ScalarType::Long ? GGML_TYPE_I32 : to_ggml_type(meta);
}

std::array<int64_t, GGML_MAX_DIMS> reversed_extents(
    const std::vector<int64_t>& sizes) {
  if (sizes.size() > GGML_MAX_DIMS) {
    throw std::runtime_error(
        "ggml: only rank 0 through 4 tensors are supported");
  }
  std::array<int64_t, GGML_MAX_DIMS> ne{1, 1, 1, 1};
  for (size_t i = 0; i < sizes.size(); ++i) {
    ne[i] = sizes[sizes.size() - 1 - i];
  }
  return ne;
}

ggml_tensor* new_tensor(
    ggml_context* ctx,
    ggml_type type,
    const std::vector<int64_t>& sizes) {
  const auto ne = reversed_extents(sizes);
  return ggml_new_tensor(
      ctx, type, std::max(1, static_cast<int>(sizes.size())), ne.data());
}

std::string op_key(std::string_view target) {
  if (target.starts_with(kTorchOpsPrefix)) {
    target.remove_prefix(kTorchOpsPrefix.size());
  }
  return std::string(target);
}

const NamedArgument*
find_arg(const Node& node, std::string_view name, size_t positional) {
  const auto named = std::find_if(
      node.inputs.begin(), node.inputs.end(), [&](const NamedArgument& arg) {
        return arg.name == name;
      });
  if (named != node.inputs.end()) {
    return &*named;
  }
  return positional < node.inputs.size() ? &node.inputs[positional] : nullptr;
}

const NamedArgument&
require_arg(const Node& node, std::string_view name, size_t positional) {
  const NamedArgument* arg = find_arg(node, name, positional);
  if (arg == nullptr) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' is missing argument '" +
        std::string(name) + "'");
  }
  return *arg;
}

std::vector<ValueId> output_ids(const Node& node) {
  std::vector<ValueId> ids;
  for (const Output& output : node.outputs) {
    if (output.kind == OutputValueKind::TensorList) {
      for (const ValueId id : output.elem_ids) {
        if (valid(id)) {
          ids.push_back(id);
        }
      }
    } else if (valid(output.value_id)) {
      ids.push_back(output.value_id);
    }
  }
  return ids;
}

std::vector<int64_t> broadcast_shape(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t x = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t y = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (x != y && x != 1 && y != 1) {
      throw std::runtime_error("ggml: incompatible broadcast shapes");
    }
    out[i] = x == 1 ? y : x;
  }
  return out;
}

int64_t normalize_dim(int64_t dim, size_t rank) {
  if (dim < 0) {
    dim += static_cast<int64_t>(rank);
  }
  if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
    throw std::runtime_error("ggml: tensor dimension is out of range");
  }
  return dim;
}

std::vector<uint8_t> narrow_to_int32(ByteSpan bytes) {
  const size_t numel = bytes.size() / sizeof(int64_t);
  std::vector<uint8_t> out(numel * sizeof(int32_t));
  const auto* src = reinterpret_cast<const int64_t*>(bytes.data());
  auto* dst = reinterpret_cast<int32_t*>(out.data());
  for (size_t i = 0; i < numel; ++i) {
    if (src[i] < std::numeric_limits<int32_t>::min() ||
        src[i] > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("ggml: int64 input does not fit in int32");
    }
    dst[i] = static_cast<int32_t>(src[i]);
  }
  return out;
}

struct TensorRef {
  ggml_tensor* tensor = nullptr;
  std::vector<int64_t> sizes;
};

// A torchao activation quantize/dequantize chain in flight; `input` is the
// float tensor it started from.
struct DynamicQuant {
  TensorRef input;
};

using RuntimeValue = std::
    variant<std::monostate, TensorRef, DynamicQuant, int64_t, double, bool>;

struct InputSlot {
  ValueId id = kInvalid;
  TensorMeta meta;
  std::vector<int64_t> sizes;
  std::vector<uint8_t> bytes;
};

struct OutputSlot {
  ValueId id = kInvalid;
  TensorMeta meta;
  std::vector<int64_t> sizes;
  // Points into the executable's output staging until the next execute.
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
};

struct Upload {
  ggml_tensor* tensor = nullptr;
  std::vector<uint8_t> bytes;
  // Signature position of the graph input these bytes come from; empty for
  // tensors that lowering fills itself, such as causal masks.
  std::optional<size_t> input;
};

struct RowIndexCheck {
  ValueId input_id = kInvalid;
  int64_t rows = 0;
};

struct GraphInstance {
  ggml_context_ptr context;
  ggml_cgraph* graph = nullptr;
  std::vector<Upload> uploads;
  std::vector<TensorRef> outputs;
  // Input-driven row indices, rechecked on every replay.
  std::vector<RowIndexCheck> row_index_checks;
};

const ggml_tensor* storage_root(const ggml_tensor* tensor) {
  while (tensor->view_src != nullptr) {
    tensor = tensor->view_src;
  }
  return tensor;
}

// ggml's row gathers and scatters do not bounds-check: a stray row aborts on
// the CPU and reads or writes out of bounds on GPUs. Input bytes are empty
// while compiling.
void check_index_bytes(const InputSlot& input, int64_t rows) {
  if (input.bytes.empty()) {
    return;
  }
  const auto in_range = [rows](int64_t row) { return row >= 0 && row < rows; };
  bool ok = true;
  if (input.meta.dtype == ScalarType::Long) {
    std::vector<int64_t> indices(input.bytes.size() / sizeof(int64_t));
    std::memcpy(indices.data(), input.bytes.data(), input.bytes.size());
    ok = std::ranges::all_of(indices, in_range);
  } else {
    std::vector<int32_t> indices(input.bytes.size() / sizeof(int32_t));
    std::memcpy(indices.data(), input.bytes.data(), input.bytes.size());
    ok = std::ranges::all_of(indices, in_range);
  }
  if (!ok) {
    throw std::out_of_range(
        "ggml: row index is outside [0, " + std::to_string(rows) + ")");
  }
}

// A small integer tensor computed on the host, contiguous and row-major.
struct HostInts {
  std::vector<int64_t> sizes;
  std::vector<int64_t> data;
};

constexpr size_t kHostEvalMaxElements = size_t{1} << 16;

// a op b for equal shapes, or with b a single value; nullopt if fn overflows.
std::optional<HostInts> broadcast_ints(
    const HostInts& a,
    const HostInts& b,
    bool (*fn)(int64_t, int64_t, int64_t*)) {
  if (a.sizes != b.sizes && b.data.size() != 1) {
    return std::nullopt;
  }
  HostInts out = a;
  for (size_t i = 0; i < out.data.size(); ++i) {
    if (fn(a.data[i], b.data[b.data.size() == 1 ? 0 : i], &out.data[i])) {
      return std::nullopt;
    }
  }
  return out;
}

// Evaluates small integer subgraphs of graph inputs and constants on the
// host, e.g. cache positions and key indices, with the current input values.
// Returns nullopt for anything outside its op set or element budget.
class HostIntEvaluator {
 public:
  using ConstantReader = std::function<std::optional<HostInts>(ValueId)>;
  // Values that read no graph input are the same on every call.
  using StaticCache = std::unordered_map<ValueId, std::optional<HostInts>>;

  HostIntEvaluator(
      const Method& method,
      const std::vector<InputSlot>& inputs,
      ConstantReader constants,
      StaticCache& static_values)
      : method_(method),
        inputs_(inputs),
        constants_(std::move(constants)),
        static_values_(static_values) {}

  // nullptr when the value is outside the op set or element budget. The
  // pointee lives as long as this evaluator (or the static cache).
  const HostInts* tensor(ValueId id) {
    if (const auto found = static_values_.find(id);
        found != static_values_.end()) {
      return found->second ? &*found->second : nullptr;
    }
    if (const auto found = memo_.find(id); found != memo_.end()) {
      // Only input-dependent values are memoized per call.
      reads_input_ = true;
      return found->second ? &*found->second : nullptr;
    }
    const bool outer_reads_input = reads_input_;
    reads_input_ = false;
    std::optional<HostInts> result;
    try {
      result = compute(id);
    } catch (const std::exception&) {
      result.reset();
    }
    if (result && result->data.size() > kHostEvalMaxElements) {
      result.reset();
    }
    auto& cache = reads_input_ ? memo_ : static_values_;
    reads_input_ = reads_input_ || outer_reads_input;
    const auto it = cache.insert_or_assign(id, std::move(result)).first;
    return it->second ? &*it->second : nullptr;
  }

 private:
  std::optional<int64_t> integer(const Argument& arg) {
    if (arg.kind() != ArgKind::Int) {
      return std::nullopt;
    }
    const IntArg& value = arg.as_int();
    if (!valid(value.id)) {
      return value.value;
    }
    const NodeId producer = method_.graph.value(value.id).producer_id;
    if (!valid(producer)) {
      return std::nullopt;
    }
    const Node& node = method_.graph.node(producer);
    const std::string op = op_key(node.target);
    if (op == "aten.sym_size.int") {
      const NamedArgument* self = find_arg(node, "self", 0);
      const NamedArgument* dim = find_arg(node, "dim", 1);
      if (self == nullptr || self->arg.kind() != ArgKind::Tensor ||
          dim == nullptr) {
        return std::nullopt;
      }
      const std::optional<std::vector<int64_t>> sizes =
          runtime_sizes(self->arg.as_tensor().id);
      const std::optional<int64_t> d = integer(dim->arg);
      if (!sizes || !d) {
        return std::nullopt;
      }
      return (*sizes)[normalize_dim(*d, sizes->size())];
    }
    return std::nullopt;
  }

  // Shapes of graph inputs are known before evaluating any value.
  std::optional<std::vector<int64_t>> runtime_sizes(ValueId id) {
    const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
    if (input != inputs_.end()) {
      reads_input_ = true;
      return input->sizes;
    }
    const HostInts* value = tensor(id);
    return value ? std::optional(value->sizes) : std::nullopt;
  }

  const HostInts*
  arg_tensor(const Node& node, std::string_view name, size_t positional) {
    const NamedArgument* arg = find_arg(node, name, positional);
    if (arg == nullptr || arg->arg.kind() != ArgKind::Tensor ||
        !valid(arg->arg.as_tensor().id)) {
      return nullptr;
    }
    return tensor(arg->arg.as_tensor().id);
  }

  std::optional<int64_t> arg_int(
      const Node& node,
      std::string_view name,
      size_t positional,
      std::optional<int64_t> fallback = std::nullopt) {
    const NamedArgument* arg = find_arg(node, name, positional);
    if (arg == nullptr || arg->arg.kind() == ArgKind::None) {
      return fallback;
    }
    return integer(arg->arg);
  }

  std::optional<HostInts> compute(ValueId id) {
    const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
    if (input != inputs_.end()) {
      reads_input_ = true;
      HostInts out{input->sizes, {}};
      if (input->meta.dtype == ScalarType::Long) {
        out.data.resize(input->bytes.size() / sizeof(int64_t));
        std::memcpy(out.data.data(), input->bytes.data(), input->bytes.size());
      } else if (input->meta.dtype == ScalarType::Int) {
        std::vector<int32_t> narrow(input->bytes.size() / sizeof(int32_t));
        std::memcpy(narrow.data(), input->bytes.data(), input->bytes.size());
        out.data.assign(narrow.begin(), narrow.end());
      } else {
        return std::nullopt;
      }
      if (out.data.size() != checked_numel(out.sizes)) {
        return std::nullopt;
      }
      return out;
    }
    const NodeId producer = method_.graph.value(id).producer_id;
    if (!valid(producer) || !method_.graph.node(producer).is_call()) {
      return constants_(id);
    }
    const Node& node = method_.graph.node(producer);
    const std::string op = op_key(node.target);
    if (op == "aten.arange.start_step" || op == "aten.arange.start" ||
        op == "aten.arange.default") {
      const bool has_start = op != "aten.arange.default";
      const std::optional<int64_t> start =
          has_start ? arg_int(node, "start", 0) : std::optional<int64_t>(0);
      const std::optional<int64_t> end =
          arg_int(node, "end", has_start ? 1 : 0);
      const std::optional<int64_t> step = op == "aten.arange.start_step"
          ? arg_int(node, "step", 2, 1)
          : std::optional<int64_t>(1);
      if (!start || !end || !step || *step <= 0 || *end < *start ||
          static_cast<size_t>((*end - *start + *step - 1) / *step) >
              kHostEvalMaxElements) {
        return std::nullopt;
      }
      HostInts out;
      for (int64_t v = *start; v < *end; v += *step) {
        out.data.push_back(v);
      }
      out.sizes = {static_cast<int64_t>(out.data.size())};
      return out;
    }
    if (op == "aten.add.Tensor" || op == "aten.sub.Tensor") {
      const HostInts* self = arg_tensor(node, "self", 0);
      if (self == nullptr || arg_int(node, "alpha", 2, 1) != 1) {
        return std::nullopt;
      }
      const NamedArgument* other = find_arg(node, "other", 1);
      HostInts scalar;
      const HostInts* rhs = nullptr;
      if (other != nullptr && other->arg.kind() == ArgKind::Tensor) {
        rhs = tensor(other->arg.as_tensor().id);
      } else if (other != nullptr) {
        if (const std::optional<int64_t> value = integer(other->arg)) {
          scalar = HostInts{{}, {*value}};
          rhs = &scalar;
        }
      }
      if (rhs == nullptr) {
        return std::nullopt;
      }
      using Fn = bool (*)(int64_t, int64_t, int64_t*);
      const Fn fn = op == "aten.add.Tensor"
          ? Fn([](int64_t a, int64_t b, int64_t* out) {
              return __builtin_add_overflow(a, b, out);
            })
          : Fn([](int64_t a, int64_t b, int64_t* out) {
              return __builtin_sub_overflow(a, b, out);
            });
      return broadcast_ints(*self, *rhs, fn);
    }
    if (op == "aten.clone.default" || op == "aten.alias.default" ||
        op == "aten.alias_copy.default" || op == "aten.contiguous.default") {
      const HostInts* self = arg_tensor(node, "self", 0);
      return self ? std::optional(*self) : std::nullopt;
    }
    if (op == "aten.unsqueeze.default" || op == "aten.unsqueeze_copy.default") {
      const HostInts* self = arg_tensor(node, "self", 0);
      std::optional<int64_t> dim = arg_int(node, "dim", 1);
      if (self == nullptr || !dim) {
        return std::nullopt;
      }
      const int64_t rank = static_cast<int64_t>(self->sizes.size()) + 1;
      if (*dim < 0) {
        *dim += rank;
      }
      if (*dim < 0 || *dim >= rank) {
        return std::nullopt;
      }
      HostInts out = *self;
      out.sizes.insert(out.sizes.begin() + *dim, 1);
      return out;
    }
    if (op == "aten.view.default" || op == "aten.view_copy.default" ||
        op == "aten.reshape.default" || op == "aten._unsafe_view.default") {
      const HostInts* self = arg_tensor(node, "self", 0);
      const NamedArgument* shape = find_arg(node, "shape", 1);
      if (self == nullptr || shape == nullptr ||
          shape->arg.kind() != ArgKind::IntList) {
        return std::nullopt;
      }
      std::vector<int64_t> sizes;
      for (size_t i = 0; i < shape->arg.as_int_list().values.size(); ++i) {
        const IntListArg& list = shape->arg.as_int_list();
        const std::optional<int64_t> v = list.ids.empty() || !valid(list.ids[i])
            ? std::optional<int64_t>(list.values[i])
            : integer(IntArg{list.values[i], list.ids[i]});
        if (!v) {
          return std::nullopt;
        }
        sizes.push_back(*v);
      }
      const auto inferred = std::ranges::find(sizes, -1);
      if (inferred != sizes.end()) {
        *inferred = 1;
        const size_t known = checked_numel(sizes);
        if (known == 0 || self->data.size() % known != 0) {
          return std::nullopt;
        }
        *inferred = static_cast<int64_t>(self->data.size() / known);
      }
      if (checked_numel(sizes) != self->data.size()) {
        return std::nullopt;
      }
      return HostInts{std::move(sizes), self->data};
    }
    return std::nullopt;
  }

  const Method& method_;
  const std::vector<InputSlot>& inputs_;
  ConstantReader constants_;
  StaticCache& static_values_;
  std::unordered_map<ValueId, std::optional<HostInts>> memo_;
  bool reads_input_ = false;
};

// An attention mask written as a comparison between a per-key index and
// per-query positions, both small integer functions of the graph inputs (the
// causal mask of a static KV cache, for instance), attends only keys below a
// host-computable extent: past it every query is masked. Attention, and the
// mask, can then cover just that key prefix exactly. Extents round up to a
// bucket so replays keep reusing one graph while the prefix grows.
constexpr int64_t kKeyExtentBucket = 256;

struct MaskExtent {
  std::string compare; // le or lt
  bool key_lhs = true;
  ValueId key_operand = kInvalid;
  ValueId query_operand = kInvalid;
  int64_t keys = 0;
  // The SDPA mask, then each value up to and including the compare output.
  std::vector<ValueId> chain;
};

// The bucketed prefix of keys that some query attends.
int64_t attended_key_extent(
    const MaskExtent& mask,
    const HostInts& key_index,
    const HostInts& query) {
  if (key_index.data.size() != static_cast<size_t>(mask.keys) ||
      query.data.empty()) {
    return mask.keys;
  }
  // Key j is attended by some query iff compare(key[j], query) holds for
  // the extreme query value; compare(query, key) is the mirrored
  // compare(key, query).
  const auto [lo, hi] = std::ranges::minmax(query.data);
  const std::string& op = mask.compare;
  const auto attended = [&](int64_t key) {
    if (op == "le") {
      return mask.key_lhs ? key <= hi : key >= lo;
    }
    return mask.key_lhs ? key < hi : key > lo;
  };
  int64_t last = -1;
  for (int64_t j = mask.keys - 1; j >= 0 && last < 0; --j) {
    if (attended(key_index.data[static_cast<size_t>(j)])) {
      last = j;
    }
  }
  const int64_t needed = std::max<int64_t>(last + 1, 1);
  return std::min(
      mask.keys,
      (needed + kKeyExtentBucket - 1) / kKeyExtentBucket * kKeyExtentBucket);
}

// Holds a method's non-empty mutable buffer keys in `owners` for the lifetime
// of its executable.
class MutableKeyClaim {
 public:
  MutableKeyClaim(GgmlMutableKeyOwners& owners, const Method& method)
      : owners_(owners) {
    for (const DataBinding& binding : method.data_bindings) {
      if (!binding.mutated || binding.key.empty()) {
        continue;
      }
      const auto owner = owners_.find(binding.key);
      if (owner != owners_.end()) {
        throw std::runtime_error(
            "ggml: mutable buffer '" + binding.key + "' of method '" +
            method.name +
            "' is already bound by a live executable of method '" +
            owner->second +
            "'; sharing mutable buffers across executables is not implemented");
      }
    }
    for (const DataBinding& binding : method.data_bindings) {
      if (binding.mutated && !binding.key.empty() &&
          owners_.try_emplace(binding.key, method.name).second) {
        keys_.push_back(binding.key);
      }
    }
  }

  MutableKeyClaim(const MutableKeyClaim&) = delete;
  MutableKeyClaim& operator=(const MutableKeyClaim&) = delete;

  ~MutableKeyClaim() {
    for (const std::string& key : keys_) {
      owners_.erase(key);
    }
  }

 private:
  GgmlMutableKeyOwners& owners_;
  std::vector<std::string> keys_;
};

ggml_tensor* reshape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes) {
  if (!ggml_is_contiguous(tensor)) {
    tensor = ggml_cont(ctx, tensor);
  }
  const auto ne = reversed_extents(sizes);
  switch (sizes.size()) {
    case 0:
    case 1:
      return ggml_reshape_1d(ctx, tensor, ne[0]);
    case 2:
      return ggml_reshape_2d(ctx, tensor, ne[0], ne[1]);
    case 3:
      return ggml_reshape_3d(ctx, tensor, ne[0], ne[1], ne[2]);
    case 4:
      return ggml_reshape_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3]);
    default:
      throw std::runtime_error("ggml: reshape supports rank 0 through 4");
  }
}

ggml_tensor* view_with_shape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes,
    size_t offset) {
  const auto ne = reversed_extents(sizes);
  switch (sizes.size()) {
    case 0:
    case 1:
      return ggml_view_1d(ctx, tensor, ne[0], offset);
    case 2:
      return ggml_view_2d(ctx, tensor, ne[0], ne[1], tensor->nb[1], offset);
    case 3:
      return ggml_view_3d(
          ctx,
          tensor,
          ne[0],
          ne[1],
          ne[2],
          tensor->nb[1],
          tensor->nb[2],
          offset);
    case 4:
      return ggml_view_4d(
          ctx,
          tensor,
          ne[0],
          ne[1],
          ne[2],
          ne[3],
          tensor->nb[1],
          tensor->nb[2],
          tensor->nb[3],
          offset);
    default:
      throw std::runtime_error("ggml: view supports rank 0 through 4");
  }
}

class GgmlExecutable final : public EngineExecutable {
 public:
  GgmlExecutable(
      const Method& method,
      const Package& package,
      const GgmlEngineOptions& options,
      size_t* materialized_constant_bytes,
      size_t* heap_copied_constant_bytes,
      GgmlMutableKeyOwners& mutable_key_owners)
      : mutable_key_claim_(mutable_key_owners, method),
        method_(method),
        package_(package),
        options_(options) {
    index_signature();
    initialize_backends();
    initialize_persistent_tensors(materialized_constant_bytes);
    index_rope_frequencies();
    index_mask_extents();
    if (heap_copied_constant_bytes != nullptr) {
      *heap_copied_constant_bytes += heap_copied_constant_bytes_;
    }
    // Rejects unsupported ops and shapes at compile time.
    build_graph(full_key_extents());
  }

  ~GgmlExecutable() override {
    for (CachedGraph& cached : cached_graphs_) {
      ggml_backend_sched_synchronize(cached.scheduler.get());
    }
    if (scheduler_) {
      ggml_backend_sched_synchronize(scheduler_.get());
      ggml_backend_sched_reset(scheduler_.get());
    }
  }

  size_t num_inputs() const override {
    return inputs_.size();
  }

  size_t num_outputs() const override {
    return outputs_.size();
  }

  std::vector<int64_t> input_sizes(size_t i) const override {
    return inputs_.at(i).sizes;
  }

  std::vector<int64_t> output_sizes(size_t i) const override {
    return outputs_.at(i).sizes;
  }

  ScalarType input_dtype(size_t i) const override {
    return inputs_.at(i).meta.dtype;
  }

  ScalarType output_dtype(size_t i) const override {
    return outputs_.at(i).meta.dtype;
  }

  void resize_input(size_t i, const std::vector<int64_t>& sizes) override {
    InputSlot& slot = inputs_.at(i);
    if (!slot.meta.accepts_sizes(sizes)) {
      throw std::invalid_argument(
          "ggml: input shape exceeds serialized bounds");
    }
    slot.sizes = sizes;
    slot.bytes.clear();
    executed_ = false;
  }

  void set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype)
      override {
    InputSlot& slot = inputs_.at(i);
    const size_t expected = checked_numel(slot.sizes);
    if (numel != expected) {
      throw std::invalid_argument(
          "ggml: input element count does not match shape");
    }
    if (src_dtype != slot.meta.dtype) {
      throw std::invalid_argument(
          "ggml: input conversion is unsupported; input dtype must match "
          "the graph");
    }
    const size_t nbytes = checked_nbytes(slot.sizes, src_dtype);
    if (nbytes != 0 && data == nullptr) {
      throw std::invalid_argument("ggml: input data is null");
    }
    slot.bytes.resize(nbytes);
    if (nbytes != 0) {
      std::memcpy(slot.bytes.data(), data, nbytes);
    }
    executed_ = false;
  }

  void execute() override {
    executed_ = false;
    for (const InputSlot& input : inputs_) {
      if (input.bytes.size() != checked_nbytes(input.sizes, input.meta.dtype)) {
        throw std::logic_error("ggml: every input must be set before execute");
      }
    }

    if (options_.max_cached_graphs == 0) {
      ggml_backend_sched_reset(scheduler_.get());
      GraphInstance instance = build_graph(evaluate_key_extents());
      allocate(scheduler_.get(), instance);
      try {
        run(scheduler_.get(), instance);
      } catch (...) {
        ggml_backend_sched_reset(scheduler_.get());
        throw;
      }
      ggml_backend_sched_reset(scheduler_.get());
    } else {
      CachedGraph& cached = cached_graph();
      try {
        run(cached.scheduler.get(), cached.instance);
      } catch (...) {
        ggml_backend_sched_synchronize(cached.scheduler.get());
        cached_graphs_.pop_back();
        throw;
      }
    }
    executed_ = true;
  }

  void get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype)
      override {
    const OutputSlot& slot = outputs_.at(i);
    if (!executed_) {
      throw std::logic_error("ggml: output requested before execute");
    }
    if (numel != checked_numel(slot.sizes)) {
      throw std::invalid_argument(
          "ggml: output element count does not match shape");
    }
    if (numel != 0 && data == nullptr) {
      throw std::invalid_argument("ggml: output destination is null");
    }
    copy_output(slot, data, dst_dtype);
  }

 private:
  struct CachedGraph {
    std::vector<std::vector<int64_t>> input_sizes;
    // Attention key extents follow input values (positions), so they are
    // part of the key along with the input shapes.
    std::vector<int64_t> key_extents;
    GraphInstance instance;
    ggml_backend_sched_ptr scheduler;
  };

  // Lowering depends on input shapes, constants and the attention key extents
  // computed from input values, so a graph built for the same key replays
  // with fresh input bytes. The most recently used graph is last.
  CachedGraph& cached_graph() {
    std::vector<int64_t> key_extents = evaluate_key_extents();
    const auto same_key = [this, &key_extents](const CachedGraph& cached) {
      if (cached.key_extents != key_extents) {
        return false;
      }
      for (size_t i = 0; i < inputs_.size(); ++i) {
        if (cached.input_sizes[i] != inputs_[i].sizes) {
          return false;
        }
      }
      return true;
    };
    const auto hit = std::ranges::find_if(cached_graphs_, same_key);
    if (hit != cached_graphs_.end()) {
      std::rotate(hit, hit + 1, cached_graphs_.end());
      refresh_inputs(cached_graphs_.back().instance);
      return cached_graphs_.back();
    }
    if (cached_graphs_.size() >= options_.max_cached_graphs) {
      ggml_backend_sched_synchronize(cached_graphs_.front().scheduler.get());
      cached_graphs_.erase(cached_graphs_.begin());
    }
    CachedGraph cached{
        .key_extents = key_extents,
        .instance = build_graph(key_extents),
        .scheduler = make_scheduler()};
    for (const InputSlot& input : inputs_) {
      cached.input_sizes.push_back(input.sizes);
    }
    allocate(cached.scheduler.get(), cached.instance);
    cached_graphs_.push_back(std::move(cached));
    return cached_graphs_.back();
  }

  void refresh_inputs(GraphInstance& instance) const {
    for (const RowIndexCheck& check : instance.row_index_checks) {
      check_index_bytes(
          *std::ranges::find(inputs_, check.input_id, &InputSlot::id),
          check.rows);
    }
    for (Upload& upload : instance.uploads) {
      if (upload.input) {
        const InputSlot& input = inputs_.at(*upload.input);
        upload.bytes = input.meta.dtype == ScalarType::Long
            ? narrow_to_int32(input.bytes)
            : input.bytes;
      }
    }
  }

  void allocate(ggml_backend_sched_t scheduler, const GraphInstance& instance) {
    // The scheduler otherwise keeps graph inputs on the CPU backend and copies
    // them into GPU splits once per consuming view.
    for (const Upload& upload : instance.uploads) {
      ggml_backend_sched_set_tensor_backend(
          scheduler, upload.tensor, primary_backend_);
    }
    if (!ggml_backend_sched_alloc_graph(scheduler, instance.graph)) {
      ggml_backend_sched_reset(scheduler);
      throw std::runtime_error("ggml: failed to allocate execution graph");
    }
  }

  void run(ggml_backend_sched_t scheduler, const GraphInstance& instance) {
    for (const Upload& upload : instance.uploads) {
      if (upload.tensor->buffer == nullptr) {
        continue;
      }
      ggml_backend_tensor_set(
          upload.tensor, upload.bytes.data(), 0, upload.bytes.size());
    }
    const ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, instance.graph);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error(
          "ggml: graph execution failed with status " +
          std::to_string(static_cast<int>(status)));
    }
    size_t staged = 0;
    for (const TensorRef& output : instance.outputs) {
      staged += GGML_PAD(ggml_nbytes(output.tensor), kStagingAlignment);
    }
    uint8_t* staging = output_staging(staged);
    for (size_t i = 0; i < outputs_.size(); ++i) {
      OutputSlot& output = outputs_[i];
      ggml_tensor* tensor = instance.outputs.at(i).tensor;
      output.sizes = instance.outputs.at(i).sizes;
      output.data = staging;
      output.nbytes = ggml_nbytes(tensor);
      ggml_backend_tensor_get(tensor, staging, 0, output.nbytes);
      staging += GGML_PAD(output.nbytes, kStagingAlignment);
    }
  }

  void index_signature() {
    for (const ValueId id : method_.graph.input_ids) {
      const TensorMeta& meta = contiguous_meta(id);
      if (meta.quant) {
        throw std::runtime_error(
            "ggml: quantized graph inputs are unsupported");
      }
      inputs_.push_back(InputSlot{.id = id, .meta = meta, .sizes = meta.sizes});
    }
    if (method_.output_specs.size() != method_.graph.output_ids.size()) {
      throw std::runtime_error("ggml: inconsistent output specifications");
    }
    for (size_t i = 0; i < method_.graph.output_ids.size(); ++i) {
      const OutputKind kind = method_.output_specs[i].kind;
      if (kind == OutputKind::UserInputMutation) {
        throw std::runtime_error("ggml: user input mutation is unsupported");
      }
      if (kind != OutputKind::UserOutput) {
        continue;
      }
      const ValueId id = method_.graph.output_ids[i];
      const TensorMeta& meta = contiguous_meta(id);
      if (meta.quant) {
        throw std::runtime_error(
            "ggml: quantized graph outputs are unsupported");
      }
      outputs_.push_back(
          OutputSlot{.id = id, .meta = meta, .sizes = meta.sizes});
    }
    for (const DataBinding& binding : method_.data_bindings) {
      if (binding.has_data && valid(binding.value_id) &&
          method_.graph.value(binding.value_id).is_tensor()) {
        contiguous_meta(binding.value_id);
      }
    }
  }

  // Bytes that cross the engine boundary are laid out as dim_order_hint says.
  const TensorMeta& contiguous_meta(ValueId id) const {
    const TensorMeta& meta = method_.graph.value(id).tensor_meta();
    if (!meta.is_contiguous()) {
      throw std::runtime_error(
          "ggml: tensor '" + method_.graph.value(id).name +
          "' has a non-contiguous dim order");
    }
    return meta;
  }

  void initialize_backends() {
    if (options_.device == GgmlDevice::Gpu) {
      ggml_backend_dev_t device = select_ggml_gpu_device(options_);
      ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
      if (!gpu) {
        throw std::runtime_error(
            std::string("ggml: failed to initialize ") +
            ggml_backend_dev_name(device));
      }
      primary_backend_ = gpu.get();
      backends_.push_back(std::move(gpu));
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!cpu) {
      throw std::runtime_error("ggml: failed to initialize CPU backend");
    }
    if (options_.cpu_threads > 0) {
      ggml_backend_cpu_set_n_threads(cpu.get(), options_.cpu_threads);
    }
    if (primary_backend_ == nullptr) {
      primary_backend_ = cpu.get();
    }
    backends_.push_back(std::move(cpu));

    if (options_.max_cached_graphs == 0) {
      scheduler_ = make_scheduler();
    }
  }

  ggml_backend_sched_ptr make_scheduler() const {
    std::vector<ggml_backend_t> raw;
    raw.reserve(backends_.size());
    for (const auto& backend : backends_) {
      raw.push_back(backend.get());
    }
    ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(
        raw.data(),
        nullptr,
        static_cast<int>(raw.size()),
        options_.graph_size,
        /*parallel=*/false,
        /*op_offload=*/true));
    if (!scheduler) {
      throw std::runtime_error("ggml: failed to create backend scheduler");
    }
    return scheduler;
  }

  // Quantization parameters are bound but only read through their weight.
  bool is_used(ValueId id) const {
    const std::vector<ValueId>& outputs = method_.graph.output_ids;
    return !method_.graph.value(id).consumer_ids.empty() ||
        std::find(outputs.begin(), outputs.end(), id) != outputs.end();
  }

  bool is_persistent(const DataBinding& binding) const {
    return valid(binding.value_id) &&
        method_.graph.value(binding.value_id).is_tensor() &&
        is_used(binding.value_id);
  }

  static size_t stored_nbytes(const TensorMeta& meta) {
    if (packed_quant(meta) != nullptr) {
      return packed_nbytes(meta);
    }
    if (affine_quant(meta) != nullptr) {
      return checked_numel(meta.sizes) / 2;
    }
    return checked_nbytes(meta.sizes, meta.dtype);
  }

  void check_q4_consumers(ValueId id, const std::string& key) const {
    for (const NodeId consumer : method_.graph.value(id).consumer_ids) {
      const Node& node = method_.graph.node(consumer);
      const std::string op = op_key(node.target);
      const NamedArgument* weight = op == "aten.linear.default"
          ? find_arg(node, "weight", 1)
          : op == "aten.embedding.default" ? find_arg(node, "weight", 0)
                                           : nullptr;
      if (weight == nullptr || weight->arg.kind() != ArgKind::Tensor ||
          weight->arg.as_tensor().id != id) {
        throw std::runtime_error(
            "ggml: quantized weight '" + key +
            "' may only feed linear or embedding weights, not '" + node.target +
            "'");
      }
    }
  }

  // Converters read some dtypes through typed pointers, so the package's bytes
  // are viewed in place only when aligned to the element size.
  std::optional<OwnedBytes> read_constant(const std::string& key) const {
    const std::optional<ConstantInfo> info = package_.constant_info(key);
    if (!info) {
      return std::nullopt;
    }
    std::optional<OwnedBytes> view = package_.view_constant(key);
    if (view &&
        reinterpret_cast<uintptr_t>(view->span().data()) %
                std::max<size_t>(element_size(info->dtype), 1) ==
            0) {
      return view;
    }
    return acquire_constant(key);
  }

  std::optional<OwnedBytes> acquire_constant(const std::string& key) const {
    std::optional<OwnedBytes> bytes = package_.acquire_constant(key);
    if (bytes) {
      heap_copied_constant_bytes_ += bytes->span().size();
    }
    return bytes;
  }

  std::vector<uint8_t> load_q4_0(const TensorMeta& meta, ByteSpan packed)
      const {
    const AffineGroupQuant& quant = q4_affine(meta);
    const std::optional<OwnedBytes> scales =
        read_constant(quant.scale_data_key);
    if (!scales) {
      throw std::runtime_error(
          "ggml: package is missing scales '" + quant.scale_data_key + "'");
    }
    if (!quant.zero_point_data_key.empty()) {
      const std::optional<OwnedBytes> zeros =
          read_constant(quant.zero_point_data_key);
      if (!zeros || std::ranges::any_of(zeros->span(), [](uint8_t byte) {
            return byte != 0;
          })) {
        throw std::runtime_error(
            "ggml: Q4_0 needs zero points of zero, see '" +
            quant.zero_point_data_key + "'");
      }
    }
    return affine_int4_to_q4_0(
        meta, packed, to_float_vector(scales->span(), quant.scale_dtype));
  }

  void initialize_persistent_tensors(size_t* materialized_constant_bytes) {
    const size_t count = static_cast<size_t>(std::ranges::count_if(
        method_.data_bindings,
        [this](const DataBinding& binding) { return is_persistent(binding); }));
    if (count == 0) {
      return;
    }
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * (count + 16),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    persistent_context_.reset(ggml_init(params));
    if (!persistent_context_) {
      throw std::runtime_error(
          "ggml: failed to create persistent tensor context");
    }

    for (const DataBinding& binding : method_.data_bindings) {
      if (!is_persistent(binding)) {
        continue;
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      if (affine_quant(meta) != nullptr || packed_quant(meta) != nullptr) {
        check_q4_consumers(binding.value_id, binding.key);
      }
      ggml_tensor* tensor =
          new_tensor(persistent_context_.get(), compute_type(meta), meta.sizes);
      ggml_set_name(tensor, binding.key.c_str());
      persistent_.emplace(binding.value_id, tensor);
    }

    persistent_buffer_.reset(ggml_backend_alloc_ctx_tensors(
        persistent_context_.get(), primary_backend_));
    if (!persistent_buffer_) {
      throw std::runtime_error("ggml: failed to allocate persistent tensors");
    }

    const bool host_buffer =
        ggml_backend_buffer_is_host(persistent_buffer_.get());
    size_t loaded = 0;
    for (const DataBinding& binding : method_.data_bindings) {
      const auto found = persistent_.find(binding.value_id);
      if (found == persistent_.end()) {
        continue;
      }
      ggml_tensor* tensor = found->second;
      if (!binding.has_data) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
        continue;
      }
      const std::optional<ConstantInfo> info =
          package_.constant_info(binding.key);
      if (!info) {
        throw std::runtime_error(
            "ggml: package is missing constant '" + binding.key + "'");
      }
      const TensorMeta& meta =
          method_.graph.value(binding.value_id).tensor_meta();
      const size_t expected = stored_nbytes(meta);
      if (info->nbytes != expected) {
        throw std::runtime_error(
            "ggml: stored byte count disagrees with PTG tensor '" +
            binding.key + "'");
      }
      const bool is_q4 = affine_quant(meta) != nullptr;
      const bool narrow =
          tensor->type == GGML_TYPE_I32 && meta.dtype == ScalarType::Long;
      if (!is_q4 && !narrow && host_buffer) {
        if (ggml_nbytes(tensor) != expected) {
          throw std::runtime_error(
              "ggml: stored byte count disagrees with ggml tensor '" +
              binding.key + "'");
        }
        package_.load_constant_into(
            binding.key,
            MutableByteSpan(static_cast<uint8_t*>(tensor->data), expected));
        loaded += expected;
        continue;
      }
      const std::optional<OwnedBytes> bytes = is_q4 || narrow
          ? read_constant(binding.key)
          // Device uploads go through a heap copy: ROCm fails to pin a
          // file-backed mapping, and ggml_backend_tensor_set from one is
          // several times slower.
          : acquire_constant(binding.key);
      if (!bytes) {
        throw std::runtime_error(
            "ggml: package is missing constant '" + binding.key + "'");
      }
      ByteSpan span = bytes->span();
      std::vector<uint8_t> converted;
      if (is_q4) {
        converted = load_q4_0(meta, span);
        span = converted;
      } else if (narrow) {
        converted = narrow_to_int32(span);
        span = converted;
      }
      if (ggml_nbytes(tensor) != span.size()) {
        throw std::runtime_error(
            "ggml: converted byte count disagrees with tensor '" + binding.key +
            "'");
      }
      ggml_backend_tensor_set(tensor, span.data(), 0, span.size());
      loaded += span.size();
    }
    ggml_backend_buffer_set_usage(
        persistent_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (materialized_constant_bytes != nullptr) {
      *materialized_constant_bytes += loaded;
    }
  }

  void index_rope_frequencies() {
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() || op_key(node.target) != "native.rope.default") {
        continue;
      }
      const Argument& arg = require_arg(node, "inv_freq", 2).arg;
      if (arg.kind() != ArgKind::Tensor) {
        throw std::runtime_error("ggml: rope inv_freq must be a tensor");
      }
      const ValueId inv_freq = arg.as_tensor().id;
      if (rope_freq_factors_.contains(inv_freq)) {
        continue;
      }
      const auto binding = std::ranges::find_if(
          method_.data_bindings, [inv_freq](const DataBinding& candidate) {
            return candidate.value_id == inv_freq && candidate.has_data;
          });
      const std::optional<OwnedBytes> bytes =
          binding == method_.data_bindings.end() ? std::nullopt
                                                 : read_constant(binding->key);
      if (!bytes) {
        throw std::runtime_error("ggml: rope inv_freq must be a constant");
      }
      std::vector<float> factors = to_float_vector(
          bytes->span(), method_.graph.value(inv_freq).tensor_meta().dtype);
      // ggml_rope_ext with freq_base 1 computes theta = pos / factor.
      for (float& factor : factors) {
        if (factor == 0.0f) {
          throw std::runtime_error("ggml: rope inv_freq has a zero entry");
        }
        factor = 1.0f / factor;
      }
      std::vector<uint8_t> raw(factors.size() * sizeof(float));
      std::memcpy(raw.data(), factors.data(), raw.size());
      rope_freq_factors_.emplace(inv_freq, std::move(raw));
    }
  }

  static void
  copy_output(const OutputSlot& slot, void* destination, ScalarType dst_dtype) {
    const size_t count = checked_numel(slot.sizes);
    const bool narrowed = slot.meta.dtype == ScalarType::Long;
    if (slot.nbytes !=
        (narrowed ? count * sizeof(int32_t)
                  : checked_nbytes(slot.sizes, slot.meta.dtype))) {
      throw std::logic_error(
          "ggml: output byte count disagrees with its dtype");
    }
    if (slot.meta.dtype == dst_dtype) {
      if (narrowed) {
        const auto* src = reinterpret_cast<const int32_t*>(slot.data);
        std::copy(src, src + count, static_cast<int64_t*>(destination));
        return;
      }
      std::memcpy(destination, slot.data, slot.nbytes);
      return;
    }
    if (dst_dtype != ScalarType::Float) {
      throw std::invalid_argument(
          "ggml: output conversion only supports conversion to Float");
    }
    float* out = static_cast<float*>(destination);
    if (slot.meta.dtype == ScalarType::Half) {
      ggml_fp16_to_fp32_row(
          reinterpret_cast<const ggml_fp16_t*>(slot.data),
          out,
          static_cast<int64_t>(count));
      return;
    }
    if (slot.meta.dtype == ScalarType::BFloat16) {
      ggml_bf16_to_fp32_row(
          reinterpret_cast<const ggml_bf16_t*>(slot.data),
          out,
          static_cast<int64_t>(count));
      return;
    }
    throw std::invalid_argument(
        "ggml: unsupported output dtype conversion to Float");
  }

  GraphInstance build_graph(const std::vector<int64_t>& key_extents);

  static std::optional<int64_t> static_dim(const TensorMeta& meta, size_t i) {
    if (i >= meta.sizes.size() ||
        (!meta.lower_bounds.empty() &&
         meta.lower_bounds.at(i) != meta.sizes[i])) {
      return std::nullopt;
    }
    return meta.sizes[i];
  }

  // Matches mask = wrappers(compare(key_index, query)) with key_index varying
  // only along the last (key) axis and query constant along it. Every value
  // in the chain must feed nothing but the next wrapper, or SDPA masks.
  std::optional<MaskExtent> match_mask_extent(ValueId mask_id, int64_t keys)
      const {
    const Graph& graph = method_.graph;
    // A boolean mask, where true means attend; additive masks never match.
    if (!graph.value(mask_id).is_tensor() ||
        graph.value(mask_id).tensor_meta().dtype != ScalarType::Bool) {
      return std::nullopt;
    }
    MaskExtent out;
    out.keys = keys;
    ValueId id = mask_id;
    for (size_t depth = 0; depth < 16; ++depth) {
      const Value& value = graph.value(id);
      if (!value.is_tensor() ||
          static_dim(value.tensor_meta(), value.tensor_meta().ndim() - 1) !=
              keys ||
          std::ranges::find(graph.output_ids, id) != graph.output_ids.end()) {
        return std::nullopt;
      }
      out.chain.push_back(id);
      const NodeId producer = value.producer_id;
      if (!valid(producer) || !graph.node(producer).is_call()) {
        return std::nullopt;
      }
      const Node& node = graph.node(producer);
      const std::string op = op_key(node.target);
      const NamedArgument* self = find_arg(node, "self", 0);
      if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
        return std::nullopt;
      }
      if (op == "aten.le.Tensor" || op == "aten.lt.Tensor") {
        const NamedArgument* other = find_arg(node, "other", 1);
        if (other == nullptr || other->arg.kind() != ArgKind::Tensor) {
          return std::nullopt;
        }
        const auto is_key = [&](ValueId operand) {
          const TensorMeta& meta = graph.value(operand).tensor_meta();
          for (size_t i = 0; i + 1 < meta.ndim(); ++i) {
            if (static_dim(meta, i) != 1) {
              return false;
            }
          }
          return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == keys;
        };
        const auto is_query = [&](ValueId operand) {
          const TensorMeta& meta = graph.value(operand).tensor_meta();
          return meta.ndim() > 0 && static_dim(meta, meta.ndim() - 1) == 1;
        };
        const ValueId a = self->arg.as_tensor().id;
        const ValueId b = other->arg.as_tensor().id;
        out.compare = op.substr(5, 2);
        if (is_key(a) && is_query(b)) {
          out.key_operand = a;
          out.query_operand = b;
        } else if (is_key(b) && is_query(a)) {
          out.key_lhs = false;
          out.key_operand = b;
          out.query_operand = a;
        } else {
          return std::nullopt;
        }
        break;
      }
      if (op == "aten.expand.default" || op == "aten.expand_copy.default") {
        // The target size of the key axis is rewritten with the extent.
        const NamedArgument* size = find_arg(node, "size", 1);
        if (size == nullptr || size->arg.kind() != ArgKind::IntList) {
          return std::nullopt;
        }
        const IntListArg& list = size->arg.as_int_list();
        if (list.values.empty() ||
            (!list.ids.empty() && valid(list.ids.back())) ||
            (list.values.back() != -1 && list.values.back() != keys)) {
          return std::nullopt;
        }
      } else if (
          op != "aten.unsqueeze.default" &&
          op != "aten.unsqueeze_copy.default" && op != "aten.alias.default" &&
          op != "aten.clone.default") {
        return std::nullopt;
      }
      id = self->arg.as_tensor().id;
    }
    if (out.compare.empty()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < out.chain.size(); ++i) {
      for (const NodeId consumer : graph.value(out.chain[i]).consumer_ids) {
        const Node& node = graph.node(consumer);
        const bool next_wrapper =
            i > 0 && graph.value(out.chain[i - 1]).producer_id == consumer;
        const NamedArgument* mask = find_arg(node, "attn_mask", 3);
        const bool sdpa_mask = i == 0 && node.is_call() &&
            op_key(node.target) ==
                "aten.scaled_dot_product_attention.default" &&
            mask != nullptr && mask->arg.kind() == ArgKind::Tensor &&
            mask->arg.as_tensor().id == out.chain[i];
        if (!next_wrapper && !sdpa_mask) {
          return std::nullopt;
        }
      }
    }
    return out;
  }

  void index_mask_extents() {
    std::unordered_map<ValueId, std::optional<MaskExtent>> by_mask;
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (!node.is_call() ||
          op_key(node.target) != "aten.scaled_dot_product_attention.default") {
        continue;
      }
      const NamedArgument* mask = find_arg(node, "attn_mask", 3);
      const NamedArgument* key = find_arg(node, "key", 1);
      if (mask == nullptr || mask->arg.kind() != ArgKind::Tensor ||
          !valid(mask->arg.as_tensor().id) || key == nullptr ||
          key->arg.kind() != ArgKind::Tensor) {
        continue;
      }
      const TensorMeta& key_meta =
          method_.graph.value(key->arg.as_tensor().id).tensor_meta();
      const std::optional<int64_t> keys = static_dim(key_meta, 2);
      const ValueId mask_id = mask->arg.as_tensor().id;
      auto [it, inserted] = by_mask.try_emplace(mask_id);
      if (inserted) {
        if (key_meta.ndim() == 4 && keys) {
          it->second = match_mask_extent(mask_id, *keys);
        }
      } else if (it->second && it->second->keys != keys) {
        // Every attention reading this mask must cover the same keys.
        it->second.reset();
      }
    }
    for (auto& [mask_id, extent] : by_mask) {
      if (extent) {
        mask_extents_.push_back(std::move(*extent));
      }
    }
  }

  std::vector<int64_t> full_key_extents() const {
    std::vector<int64_t> out;
    for (const MaskExtent& mask : mask_extents_) {
      out.push_back(mask.keys);
    }
    return out;
  }

  std::vector<int64_t> evaluate_key_extents() {
    if (mask_extents_.empty()) {
      return full_key_extents();
    }
    HostIntEvaluator host(
        method_,
        inputs_,
        [this](ValueId id) { return host_constant(id); },
        host_static_values_);
    std::vector<int64_t> out;
    for (const MaskExtent& mask : mask_extents_) {
      const HostInts* key_index = host.tensor(mask.key_operand);
      const HostInts* query = host.tensor(mask.query_operand);
      out.push_back(
          key_index != nullptr && query != nullptr
              ? attended_key_extent(mask, *key_index, *query)
              : mask.keys);
    }
    return out;
  }

  // Small integer constants, e.g. a key offset, read once from the package.
  std::optional<HostInts> host_constant(ValueId id) {
    if (const auto found = host_constants_.find(id);
        found != host_constants_.end()) {
      return found->second;
    }
    std::optional<HostInts> out;
    const auto binding = std::ranges::find_if(
        method_.data_bindings, [id](const DataBinding& candidate) {
          return candidate.value_id == id && candidate.has_data &&
              !candidate.mutated;
        });
    const TensorMeta* meta = method_.graph.value(id).is_tensor()
        ? &method_.graph.value(id).tensor_meta()
        : nullptr;
    if (binding != method_.data_bindings.end() && meta != nullptr &&
        !meta->quant && checked_numel(meta->sizes) <= 64 &&
        (meta->dtype == ScalarType::Long || meta->dtype == ScalarType::Int)) {
      if (const std::optional<OwnedBytes> bytes = read_constant(binding->key)) {
        out = HostInts{meta->sizes, {}};
        const ByteSpan span = bytes->span();
        if (meta->dtype == ScalarType::Long) {
          out->data.resize(span.size() / sizeof(int64_t));
          std::memcpy(out->data.data(), span.data(), span.size());
        } else {
          std::vector<int32_t> narrow(span.size() / sizeof(int32_t));
          std::memcpy(narrow.data(), span.data(), span.size());
          out->data.assign(narrow.begin(), narrow.end());
        }
        if (out->data.size() != checked_numel(meta->sizes)) {
          out.reset();
        }
      }
    }
    host_constants_.emplace(id, out);
    return out;
  }

  static constexpr size_t kStagingAlignment = 64;

  // Device-to-host copies into pinned memory skip the driver's pageable
  // bounce buffer. Devices without a host buffer type use plain memory.
  uint8_t* output_staging(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(primary_backend_));
    if (host == nullptr) {
      output_host_.resize(std::max(output_host_.size(), bytes));
      return output_host_.data();
    }
    if (!output_staging_ ||
        ggml_backend_buffer_get_size(output_staging_.get()) < bytes) {
      output_staging_.reset(ggml_backend_buft_alloc_buffer(host, bytes));
      if (!output_staging_) {
        throw std::runtime_error("ggml: failed to allocate output staging");
      }
    }
    return static_cast<uint8_t*>(
        ggml_backend_buffer_get_base(output_staging_.get()));
  }

  MutableKeyClaim mutable_key_claim_;
  // Accumulated while compiling, reported to the context at the end.
  mutable size_t heap_copied_constant_bytes_ = 0;
  const Method& method_;
  const Package& package_;
  GgmlEngineOptions options_;
  std::vector<InputSlot> inputs_;
  std::vector<OutputSlot> outputs_;
  std::vector<ggml_backend_ptr> backends_;
  ggml_backend_t primary_backend_ = nullptr;
  ggml_backend_sched_ptr scheduler_;
  std::vector<CachedGraph> cached_graphs_;
  std::vector<MaskExtent> mask_extents_;
  std::unordered_map<ValueId, std::optional<HostInts>> host_constants_;
  HostIntEvaluator::StaticCache host_static_values_;
  ggml_context_ptr persistent_context_;
  ggml_backend_buffer_ptr persistent_buffer_;
  ggml_backend_buffer_ptr output_staging_;
  std::vector<uint8_t> output_host_;
  std::unordered_map<ValueId, ggml_tensor*> persistent_;
  bool executed_ = false;
  std::unordered_map<ValueId, std::vector<uint8_t>> rope_freq_factors_;
};

class GraphBuilder {
 public:
  GraphBuilder(
      const Method& method,
      const std::vector<InputSlot>& inputs,
      const std::vector<OutputSlot>& outputs,
      const std::unordered_map<ValueId, ggml_tensor*>& persistent,
      const std::unordered_map<ValueId, std::vector<uint8_t>>&
          rope_factor_bytes,
      std::unordered_map<ValueId, int64_t> key_extents,
      size_t graph_size)
      : method_(method),
        inputs_(inputs),
        outputs_(outputs),
        persistent_(persistent),
        rope_factor_bytes_(rope_factor_bytes),
        key_extents_(std::move(key_extents)),
        graph_size_(graph_size),
        values_(method.graph.values.size()) {
    for (const auto& [id, tensor] : persistent) {
      persistent_storage_.insert(tensor);
    }
  }

  GraphInstance build() {
    // Metadata room for the tensors lowering creates: a generous allowance per
    // value and per node for views, casts and shape helpers, plus a reserve for
    // per-graph tensors such as masks. ggml aborts if a graph outgrows it.
    constexpr size_t kTensorsPerValue = 12;
    constexpr size_t kTensorsPerNode = 8;
    constexpr size_t kReservedTensors = 256;
    const size_t tensor_budget =
        method_.graph.values.size() * kTensorsPerValue +
        method_.graph.nodes.size() * kTensorsPerNode + kReservedTensors;
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * tensor_budget +
            ggml_graph_overhead_custom(graph_size_, false),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    instance_.context.reset(ggml_init(params));
    if (!instance_.context) {
      throw std::runtime_error("ggml: failed to allocate graph metadata");
    }
    instance_.graph = ggml_new_graph_custom(
        instance_.context.get(), graph_size_, /*grads=*/false);
    if (instance_.graph == nullptr) {
      throw std::runtime_error("ggml: failed to create cgraph");
    }

    seed_values();
    index_written_values();
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      if (node.is_call()) {
        copy_required_ = touches_written_value(node);
        lower(node);
      }
    }
    // In-place writes are already in the graph (expand_write); write-backs run
    // after every output.
    for (const OutputSlot& output : outputs_) {
      TensorRef value = tensor(output.id);
      value.tensor = as_output(value.tensor, output.meta);
      ggml_set_output(value.tensor);
      ggml_build_forward_expand(instance_.graph, value.tensor);
      instance_.outputs.push_back(value);
    }
    std::vector<ggml_tensor*> copies;
    for (size_t i = 0; i < method_.output_specs.size(); ++i) {
      if (method_.output_specs[i].kind == OutputKind::BufferMutation) {
        if (ggml_tensor* copy = write_back(
                method_.graph.output_ids[i],
                method_.output_specs[i].target_id)) {
          copies.push_back(copy);
        }
      }
    }
    for (ggml_tensor* copy : copies) {
      ggml_build_forward_expand(instance_.graph, copy);
    }
    return std::move(instance_);
  }

 private:
  // Outputs leave through a byte copy, so they must be contiguous and of the
  // declared type. int64 is widened on the host.
  ggml_tensor* as_output(ggml_tensor* result, const TensorMeta& meta) {
    const ggml_type type = compute_type(meta);
    if (result->type != type) {
      return ggml_cast(instance_.context.get(), result, type);
    }
    return ggml_is_contiguous(result)
        ? result
        : ggml_cont(instance_.context.get(), result);
  }

  // ggml runs nodes in the order they enter the graph. An in-place write enters
  // as soon as it is lowered, after every earlier op that reads the storage it
  // writes, so those reads see the old value as the schedule says.
  void expand_write(ggml_tensor* write) {
    const ggml_tensor* root = storage_root(write);
    ggml_context* ctx = instance_.context.get();
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t)) {
      if (t == write || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW ||
          t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
          t->op == GGML_OP_TRANSPOSE) {
        continue;
      }
      if (std::ranges::any_of(t->src, [root](const ggml_tensor* src) {
            return src != nullptr && storage_root(src) == root;
          })) {
        ggml_build_forward_expand(instance_.graph, t);
      }
    }
    ggml_build_forward_expand(instance_.graph, write);
  }

  // In-place lowerings already wrote their buffer; anything else is copied
  // back after every output has read the old value. The returned copy must
  // enter the graph after every write-back's value, so that no value reads a
  // buffer another write-back has already overwritten.
  ggml_tensor* write_back(ValueId value_id, ValueId target_id) {
    const auto target = persistent_.find(target_id);
    if (target == persistent_.end()) {
      throw std::runtime_error(
          "ggml: mutated buffer is not a persistent tensor");
    }
    ggml_tensor* value = tensor(value_id).tensor;
    if (value == target->second || value->view_src == target->second) {
      return nullptr;
    }
    value = snapshot(value);
    ggml_build_forward_expand(instance_.graph, value);
    return ggml_cpy(instance_.context.get(), value, target->second);
  }

  // `tensor`, or a copy of it if it shares persistent storage, which
  // write-backs and in-place writes may overwrite.
  ggml_tensor* snapshot(ggml_tensor* tensor) {
    const ggml_tensor* root = storage_root(tensor);
    for (const auto& [id, persistent] : persistent_) {
      if (persistent == root) {
        return ggml_cont(instance_.context.get(), tensor);
      }
    }
    return tensor;
  }

  void seed_values() {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const InputSlot& input = inputs_[i];
      const bool narrow = input.meta.dtype == ScalarType::Long;
      ggml_tensor* tensor = new_tensor(
          instance_.context.get(), compute_type(input.meta), input.sizes);
      ggml_set_name(tensor, method_.graph.value(input.id).name.c_str());
      ggml_set_input(tensor);
      set(input.id, TensorRef{tensor, input.sizes});
      instance_.uploads.push_back(
          Upload{
              .tensor = tensor,
              .bytes = narrow ? narrow_to_int32(input.bytes) : input.bytes,
              .input = i});
    }
    for (const auto& [id, tensor] : persistent_) {
      set(id, TensorRef{tensor, method_.graph.value(id).tensor_meta().sizes});
    }
    for (ValueId id = 0; id < static_cast<ValueId>(method_.graph.values.size());
         ++id) {
      const Value& value = method_.graph.value(id);
      if (!value.is_scalar()) {
        continue;
      }
      const Scalar& scalar = value.scalar();
      if (scalar.is_int()) {
        set(id, scalar.to_int());
      } else if (scalar.is_double()) {
        set(id, scalar.to_double());
      } else {
        set(id, scalar.to_bool());
      }
    }
  }

  void set(ValueId id, RuntimeValue value) {
    if (!in_bounds(id, values_.size())) {
      throw std::runtime_error("ggml: value id is out of range");
    }
    const auto* ref = std::get_if<TensorRef>(&value);
    if (ref != nullptr) {
      const auto ne = reversed_extents(ref->sizes);
      if (!std::equal(ne.begin(), ne.end(), ref->tensor->ne)) {
        throw std::logic_error(
            "ggml: lowered tensor shape disagrees with value '" +
            method_.graph.value(id).name + "'");
      }
    }
    values_.at(static_cast<size_t>(id)) = std::move(value);
  }

  const RuntimeValue& value(ValueId id) const {
    if (!in_bounds(id, values_.size())) {
      throw std::runtime_error("ggml: argument value id is out of range");
    }
    return values_.at(static_cast<size_t>(id));
  }

  // The value as lowered, which may be a strided view (see set_view).
  TensorRef view(ValueId id) const {
    const TensorRef* ref = std::get_if<TensorRef>(&value(id));
    if (ref == nullptr || ref->tensor == nullptr) {
      throw std::runtime_error("ggml: argument does not resolve to a tensor");
    }
    return *ref;
  }

  // The value with any deferred permutation copied out, once per value.
  TensorRef tensor(ValueId id) const {
    TensorRef ref = view(id);
    if (!deferred_.contains(id)) {
      return ref;
    }
    auto [it, inserted] = deferred_copies_.try_emplace(id, nullptr);
    if (inserted) {
      it->second = ggml_cont(instance_.context.get(), ref.tensor);
    }
    ref.tensor = it->second;
    return ref;
  }

  TensorRef
  tensor_arg(const Node& node, std::string_view name, size_t positional) const {
    return tensor(tensor_arg_id(node, name, positional));
  }

  // For lowerings whose ggml ops read any strides.
  TensorRef view_arg(const Node& node, std::string_view name, size_t positional)
      const {
    return view(tensor_arg_id(node, name, positional));
  }

  // For lowerings whose ggml ops read any outer strides but need contiguous
  // rows: a deferred permutation that moves the last dim is copied.
  TensorRef rows_arg(const Node& node, std::string_view name, size_t positional)
      const {
    const ValueId id = tensor_arg_id(node, name, positional);
    const TensorRef ref = view(id);
    return ggml_is_contiguous_rows(ref.tensor) ? ref : tensor(id);
  }

  ValueId tensor_arg_id(
      const Node& node,
      std::string_view name,
      size_t positional) const {
    const Argument& arg = require_arg(node, name, positional).arg;
    if (arg.kind() != ArgKind::Tensor || !valid(arg.as_tensor().id)) {
      throw std::runtime_error(
          "ggml: argument '" + std::string(name) + "' must be a tensor");
    }
    return arg.as_tensor().id;
  }

  // The result of a permutation. A contiguous result goes through
  // materialize(). A strided view of storage the graph never writes stays a
  // view, shared like materialize()'s (so a later in-place write throws):
  // generic consumers get one copy from tensor(), while view-aware lowerings
  // (permute, cast, rope, index_put rows, attention, matmul's right operand)
  // read the strides, so a transpose -> rope -> transpose back chain costs no
  // copies at any length.
  void set_view(ValueId id, TensorRef ref) {
    const ggml_tensor* root = storage_root(ref.tensor);
    if (ggml_is_contiguous(ref.tensor)) {
      ref.tensor = materialize(ref.tensor);
      set(id, std::move(ref));
    } else if (
        copy_required_ || persistent_storage_.contains(root) ||
        written_storage_.contains(root)) {
      ref.tensor = ggml_cont(instance_.context.get(), ref.tensor);
      set(id, std::move(ref));
    } else {
      aliased_storage_.insert(root);
      set(id, std::move(ref));
      deferred_.insert(id);
    }
  }

  void set_single_view(const Node& node, TensorRef result) {
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error(
          "ggml: op '" + node.target + "' must produce one tensor");
    }
    set_view(ids.front(), std::move(result));
  }

  // The key extent a mask value is computed over, if truncated.
  std::optional<int64_t> key_extent_of(const Node& node) const {
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      return std::nullopt;
    }
    const auto found = key_extents_.find(ids.front());
    return found == key_extents_.end() ? std::nullopt
                                       : std::optional(found->second);
  }

  // The first `extent` entries along the last axis.
  TensorRef last_axis_prefix(TensorRef ref, int64_t extent) {
    if (ref.sizes.empty() || ref.sizes.back() <= extent) {
      return ref;
    }
    ggml_tensor* t = ref.tensor;
    ref.tensor = ggml_view_4d(
        instance_.context.get(),
        t,
        extent,
        t->ne[1],
        t->ne[2],
        t->ne[3],
        t->nb[1],
        t->nb[2],
        t->nb[3],
        0);
    ref.sizes.back() = extent;
    return ref;
  }

  int64_t int_value(const Argument& arg) const {
    if (arg.kind() != ArgKind::Int) {
      throw std::runtime_error("ggml: expected integer argument");
    }
    const IntArg& integer = arg.as_int();
    if (!valid(integer.id)) {
      return integer.value;
    }
    const RuntimeValue& resolved = value(integer.id);
    if (const auto* i = std::get_if<int64_t>(&resolved)) {
      return *i;
    }
    throw std::runtime_error("ggml: symbolic integer is unresolved");
  }

  int64_t int_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      int64_t fallback,
      bool optional = false) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() == ArgKind::None) {
      if (optional) {
        return fallback;
      }
      throw std::runtime_error(
          "ggml: op '" + node.target + "' is missing integer argument '" +
          std::string(name) + "'");
    }
    return int_value(found->arg);
  }

  double float_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      double fallback,
      bool optional = false) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() == ArgKind::None) {
      if (optional) {
        return fallback;
      }
      throw std::runtime_error("ggml: missing floating-point argument");
    }
    if (found->arg.kind() == ArgKind::Float) {
      const FloatArg& arg = found->arg.as_float();
      if (!valid(arg.id)) {
        return arg.value;
      }
      const RuntimeValue& resolved = value(arg.id);
      if (const auto* f = std::get_if<double>(&resolved)) {
        return *f;
      }
    }
    if (found->arg.kind() == ArgKind::Int) {
      return static_cast<double>(int_value(found->arg));
    }
    throw std::runtime_error("ggml: floating-point argument is unresolved");
  }

  bool bool_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      bool fallback) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() == ArgKind::None) {
      return fallback;
    }
    if (found->arg.kind() != ArgKind::Bool) {
      throw std::runtime_error("ggml: expected boolean argument");
    }
    const BoolArg& arg = found->arg.as_bool();
    if (!valid(arg.id)) {
      return arg.value;
    }
    const RuntimeValue& resolved = value(arg.id);
    if (const auto* b = std::get_if<bool>(&resolved)) {
      return *b;
    }
    throw std::runtime_error("ggml: symbolic boolean is unresolved");
  }

  std::vector<int64_t> int_list(const Argument& arg) const {
    if (arg.kind() != ArgKind::IntList) {
      throw std::runtime_error("ggml: expected integer-list argument");
    }
    const IntListArg& list = arg.as_int_list();
    std::vector<int64_t> out = list.values;
    if (list.ids.empty()) {
      return out;
    }
    if (list.ids.size() != out.size()) {
      throw std::runtime_error("ggml: malformed symbolic integer list");
    }
    for (size_t i = 0; i < out.size(); ++i) {
      if (valid(list.ids[i])) {
        const RuntimeValue& resolved = value(list.ids[i]);
        const auto* integer = std::get_if<int64_t>(&resolved);
        if (integer == nullptr) {
          throw std::runtime_error("ggml: symbolic list element is unresolved");
        }
        out[i] = *integer;
      }
    }
    return out;
  }

  void set_single_output(const Node& node, TensorRef result) {
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error(
          "ggml: op '" + node.target + "' must produce one tensor");
    }
    set(ids.front(), std::move(result));
  }

  void lower(const Node& node) {
    const std::string key = op_key(node.target);
    if (key == "aten.linear.default") {
      lower_linear(node);
    } else if (key == "aten.embedding.default") {
      lower_embedding(node);
    } else if (key == "aten.rms_norm.default") {
      lower_rms_norm(node);
    } else if (
        key == "aten.view_copy.default" || key == "aten.reshape.default" ||
        key == "aten._unsafe_view.default" || key == "aten.view.default") {
      lower_reshape(node);
    } else if (
        key == "aten.permute_copy.default" || key == "aten.permute.default") {
      lower_permute(node);
    } else if (key == "aten.transpose_copy.int") {
      lower_transpose(node);
    } else if (
        key == "aten.clone.default" || key == "aten.contiguous.default" ||
        key == "aten.alias.default") {
      lower_copy_or_alias(node, key == "aten.alias.default");
    } else if (key == "aten._to_copy.default") {
      lower_cast(node);
    } else if (
        key == "aten.add.Tensor" || key == "aten.mul.Tensor" ||
        key == "aten.sub.Tensor" || key == "aten.div.Tensor") {
      lower_binary(node, key);
    } else if (key == "aten.silu.default") {
      lower_unary(node, ggml_silu);
    } else if (
        key == "aten.sigmoid.default" || key == "aten.sigmoid_.default") {
      lower_unary(node, ggml_sigmoid);
    } else if (key == "aten.gelu.default" || key == "aten.gelu_.default") {
      lower_gelu(node);
    } else if (key == "aten.native_layer_norm.default") {
      lower_layer_norm(node);
    } else if (key == "aten.constant_pad_nd.default") {
      lower_constant_pad(node);
    } else if (key == "aten.log.default") {
      lower_unary(node, ggml_log);
    } else if (key == "aten.neg.default") {
      lower_unary(node, ggml_neg);
    } else if (key == "aten.relu.default" || key == "aten.relu_.default") {
      lower_unary(node, ggml_relu);
    } else if (
        key == "aten.hardtanh.default" || key == "aten.hardtanh_.default") {
      lower_hardtanh(node);
    } else if (key == "aten.convolution.default") {
      lower_convolution(node);
    } else if (key == "aten._native_batch_norm_legit_no_training.default") {
      lower_batch_norm(node);
    } else if (
        key == "aten.max_pool2d_with_indices.default" ||
        key == "aten.max_pool2d.default") {
      lower_max_pool2d(node);
    } else if (key == "aten.mean.dim") {
      lower_mean(node);
    } else if (key == "aten.slice_copy.Tensor" || key == "aten.slice.Tensor") {
      lower_slice(node, key == "aten.slice_copy.Tensor");
    } else if (key == "aten.select_copy.int" || key == "aten.select.int") {
      lower_select(node, key == "aten.select_copy.int");
    } else if (
        key == "aten.unsqueeze_copy.default" ||
        key == "aten.unsqueeze.default") {
      lower_unsqueeze(node);
    } else if (key == "aten.squeeze_copy.dims" || key == "aten.squeeze.dims") {
      lower_squeeze(node);
    } else if (
        key == "aten.expand_copy.default" || key == "aten.expand.default") {
      lower_expand(node);
    } else if (key == "aten.cat.default") {
      lower_cat(node);
    } else if (
        key == "aten.matmul.default" || key == "aten.mm.default" ||
        key == "aten.bmm.default") {
      lower_matmul(node);
    } else if (key == "llama.update_cache.default") {
      lower_update_cache(node);
    } else if (key == "llama.custom_sdpa.default") {
      lower_custom_sdpa(node);
    } else if (key == "aten.le.Tensor" || key == "aten.le.Scalar") {
      lower_compare(node, /*strict=*/false);
    } else if (key == "aten.lt.Tensor" || key == "aten.lt.Scalar") {
      lower_compare(node, /*strict=*/true);
    } else if (key == "aten.arange.start_step") {
      lower_arange(node);
    } else if (key == "torchao.choose_qparams_affine.default") {
      lower_choose_qparams(node);
    } else if (key == "torchao.quantize_affine.default") {
      lower_quantize_affine(node);
    } else if (key == "torchao.dequantize_affine.default") {
      lower_dequantize_affine(node);
    } else if (key == "native.rope.default") {
      lower_rope(node);
    } else if (key == "aten.index_put_.default") {
      lower_index_put(node);
    } else if (key == "aten.scaled_dot_product_attention.default") {
      lower_sdpa(node);
    } else if (key == "aten.sym_size.int") {
      lower_sym_size(node);
    } else if (
        key == "_operator.add" || key == "_operator.sub" ||
        key == "_operator.mul" || key == "_operator.floordiv") {
      lower_symbolic(node, key);
    } else {
      throw std::runtime_error(
          "ggml: unsupported PTG op '" + node.target + "'");
    }
  }

  void lower_linear(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const TensorRef weight = tensor_arg(node, "weight", 1);
    if (weight.sizes.size() != 2 || input.sizes.empty() ||
        input.sizes.back() != weight.sizes[1]) {
      throw std::runtime_error("ggml: invalid linear shapes");
    }
    // ggml-cpu multiplies quantized or mixed-type weights only by F32
    // activations. mul_mat returns F32 whatever its operand types.
    ggml_tensor* activation = input.tensor->type == GGML_TYPE_F32 ||
            input.tensor->type == weight.tensor->type
        ? input.tensor
        : as_f32(input.tensor);
    ggml_tensor* result =
        ggml_mul_mat(instance_.context.get(), weight.tensor, activation);
    if (const std::optional<TensorRef> bias =
            optional_tensor(node, "bias", 2)) {
      if (bias->sizes != std::vector<int64_t>{weight.sizes[0]}) {
        throw std::runtime_error(
            "ggml: linear bias must hold one value per output feature");
      }
      result = ggml_add(instance_.context.get(), result, as_f32(bias->tensor));
    }
    std::vector<int64_t> sizes = input.sizes;
    sizes.back() = weight.sizes[0];
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_embedding(const Node& node) {
    const TensorRef weight = tensor_arg(node, "weight", 0);
    TensorRef indices = tensor_arg(node, "indices", 1);
    check_row_indices(
        require_arg(node, "indices", 1).arg.as_tensor().id, weight.sizes.at(0));
    if (indices.tensor->type != GGML_TYPE_I32) {
      indices.tensor =
          ggml_cast(instance_.context.get(), indices.tensor, GGML_TYPE_I32);
    }
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* flat =
        reshape(ctx, indices.tensor, {ggml_nelements(indices.tensor)});
    std::vector<int64_t> sizes = indices.sizes;
    sizes.push_back(weight.sizes.at(1));
    ggml_tensor* result =
        reshape(ctx, ggml_get_rows(ctx, weight.tensor, flat), sizes);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  // Row indices are checked on the host, which needs them to come from a
  // graph input, possibly through integer casts and views; any other source
  // is rejected.
  void check_row_indices(ValueId id, int64_t rows) {
    static const std::unordered_set<std::string> kSameValues = {
        "aten._to_copy.default",
        "aten.alias.default",
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.expand.default",
        "aten.expand_copy.default",
        "aten.permute.default",
        "aten.permute_copy.default",
        "aten.reshape.default",
        "aten._unsafe_view.default",
        "aten.select.int",
        "aten.select_copy.int",
        "aten.slice.Tensor",
        "aten.slice_copy.Tensor",
        "aten.squeeze.dims",
        "aten.squeeze_copy.dims",
        "aten.transpose_copy.int",
        "aten.unsqueeze.default",
        "aten.unsqueeze_copy.default",
        "aten.view.default",
        "aten.view_copy.default",
    };
    const auto is_index = [this](ValueId value) {
      const ScalarType dtype = method_.graph.value(value).tensor_meta().dtype;
      return dtype == ScalarType::Long || dtype == ScalarType::Int;
    };
    while (is_index(id)) {
      const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
      if (input != inputs_.end()) {
        check_index_bytes(*input, rows);
        instance_.row_index_checks.push_back(
            RowIndexCheck{.input_id = id, .rows = rows});
        return;
      }
      const NodeId producer = method_.graph.value(id).producer_id;
      if (!valid(producer)) {
        break;
      }
      const Node& node = method_.graph.node(producer);
      const NamedArgument* source = find_arg(node, "self", 0);
      if (!node.is_call() || !kSameValues.contains(op_key(node.target)) ||
          source == nullptr || source->arg.kind() != ArgKind::Tensor) {
        break;
      }
      id = source->arg.as_tensor().id;
    }
    throw std::runtime_error(
        "ggml: row indices must come from a graph input, through integer "
        "casts and views only");
  }

  void lower_rms_norm(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const std::vector<int64_t> normalized =
        int_list(require_arg(node, "normalized_shape", 1).arg);
    if (input.sizes.empty() || normalized.size() != 1 ||
        normalized[0] != input.sizes.back()) {
      throw std::runtime_error("ggml: rms_norm only normalizes the last dim");
    }
    const double eps =
        float_arg(node, "eps", 3, std::numeric_limits<float>::epsilon(), true);
    // ggml's norm kernels only read F32.
    ggml_tensor* result = ggml_rms_norm(
        instance_.context.get(), as_f32(input.tensor), static_cast<float>(eps));
    if (const std::optional<TensorRef> weight =
            optional_tensor(node, "weight", 2)) {
      result =
          ggml_mul(instance_.context.get(), result, as_f32(weight->tensor));
    }
    set_single_output(node, TensorRef{result, input.sizes});
  }

  void lower_layer_norm(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const std::vector<int64_t> normalized =
        int_list(require_arg(node, "normalized_shape", 1).arg);
    if (input.sizes.empty() || normalized.size() != 1 ||
        normalized[0] != input.sizes.back()) {
      throw std::runtime_error("ggml: layer_norm only normalizes the last dim");
    }
    require_only_first_output_used(node);
    // ggml's norm kernels only read F32.
    require_f32(node, input.tensor);
    const double eps = float_arg(node, "eps", 4, 1e-5, true);
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* result =
        ggml_norm(ctx, contiguous(input.tensor), static_cast<float>(eps));
    if (const std::optional<TensorRef> weight =
            optional_tensor(node, "weight", 2)) {
      result = ggml_mul(ctx, result, weight->tensor);
    }
    if (const std::optional<TensorRef> bias =
            optional_tensor(node, "bias", 3)) {
      result = ggml_add(ctx, result, bias->tensor);
    }
    set(output_ids(node).at(0), TensorRef{result, input.sizes});
  }

  void lower_gelu(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    require_float(node, input.tensor);
    const NamedArgument* approximate = find_arg(node, "approximate", 1);
    const bool tanh = approximate != nullptr &&
        approximate->arg.kind() == ArgKind::String &&
        approximate->arg.as_string().value == "tanh";
    ggml_context* ctx = instance_.context.get();
    set_single_output(
        node,
        TensorRef{
            tanh ? ggml_gelu(ctx, input.tensor)
                 : ggml_gelu_erf(ctx, input.tensor),
            input.sizes});
  }

  // pad lists (left, right) pairs from the last dim inward, which is ggml's
  // ne order.
  void lower_constant_pad(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    const std::vector<int64_t> pad = int_list(require_arg(node, "pad", 1).arg);
    const size_t rank = input.sizes.size();
    if (pad.size() % 2 != 0 || pad.size() / 2 > rank || rank > 4) {
      throw std::runtime_error("ggml: invalid constant_pad_nd padding");
    }
    std::array<int, 8> amounts{};
    std::vector<int64_t> sizes = input.sizes;
    for (size_t i = 0; i < pad.size(); ++i) {
      if (pad[i] < 0) {
        throw std::runtime_error("ggml: constant_pad_nd cannot crop");
      }
      amounts[i] = static_cast<int>(pad[i]);
      sizes[rank - 1 - i / 2] += pad[i];
    }
    if (std::ranges::all_of(amounts, [](int p) { return p == 0; })) {
      set_single_output(
          node, TensorRef{materialize(input.tensor), input.sizes});
      return;
    }
    if (float_arg(node, "value", 2, 0.0, true) != 0.0) {
      throw std::runtime_error("ggml: constant_pad_nd only pads with zeros");
    }
    require_f32(node, input.tensor);
    ggml_tensor* result = ggml_pad_ext(
        instance_.context.get(),
        contiguous(input.tensor),
        amounts[0],
        amounts[1],
        amounts[2],
        amounts[3],
        amounts[4],
        amounts[5],
        amounts[6],
        amounts[7]);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_reshape(const Node& node) {
    TensorRef input = tensor_arg(node, "self", 0);
    if (op_key(node.target) == "aten.view_copy.default") {
      input.tensor = snapshot(input.tensor);
    }
    std::vector<int64_t> sizes = int_list(require_arg(node, "shape", 1).arg);
    int64_t inferred = -1;
    size_t known = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] == -1) {
        if (inferred != -1) {
          throw std::runtime_error("ggml: reshape has multiple inferred dims");
        }
        inferred = static_cast<int64_t>(i);
      } else if (sizes[i] < 0) {
        throw std::runtime_error("ggml: reshape has invalid dimension");
      } else {
        known *= static_cast<size_t>(sizes[i]);
      }
    }
    if (inferred >= 0) {
      const size_t total = checked_numel(input.sizes);
      if (known == 0 || total % known != 0) {
        throw std::runtime_error("ggml: reshape cannot infer dimension");
      }
      sizes[static_cast<size_t>(inferred)] =
          static_cast<int64_t>(total / known);
    }
    if (checked_numel(sizes) != checked_numel(input.sizes)) {
      throw std::runtime_error("ggml: reshape changes the element count");
    }
    set_single_output(
        node,
        TensorRef{
            reshape(instance_.context.get(), input.tensor, sizes), sizes});
  }

  void lower_permute(const Node& node) {
    const TensorRef input = view_arg(node, "self", 0);
    const std::vector<int64_t> dims =
        int_list(require_arg(node, "dims", 1).arg);
    if (dims.size() != input.sizes.size() || dims.size() > 4) {
      throw std::runtime_error("ggml: invalid permutation rank");
    }
    std::array<int, 4> axes{0, 1, 2, 3};
    std::vector<int64_t> sizes(dims.size());
    std::vector<bool> seen(dims.size());
    for (size_t out_dim = 0; out_dim < dims.size(); ++out_dim) {
      const int64_t src_dim = normalize_dim(dims[out_dim], dims.size());
      if (seen[static_cast<size_t>(src_dim)]) {
        throw std::runtime_error("ggml: repeated permutation dimension");
      }
      seen[static_cast<size_t>(src_dim)] = true;
      sizes[out_dim] = input.sizes[static_cast<size_t>(src_dim)];
    }
    // ggml_permute maps source axis i to result axis axes[i].
    for (size_t out_dim = 0; out_dim < dims.size(); ++out_dim) {
      const int64_t src_dim = normalize_dim(dims[out_dim], dims.size());
      axes[dims.size() - 1 - static_cast<size_t>(src_dim)] =
          static_cast<int>(dims.size() - 1 - out_dim);
    }
    ggml_tensor* result = ggml_permute(
        instance_.context.get(),
        input.tensor,
        axes[0],
        axes[1],
        axes[2],
        axes[3]);
    set_single_view(node, TensorRef{result, std::move(sizes)});
  }

  void lower_transpose(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    const int64_t d0 =
        normalize_dim(int_arg(node, "dim0", 1, 0), input.sizes.size());
    const int64_t d1 =
        normalize_dim(int_arg(node, "dim1", 2, 0), input.sizes.size());
    std::vector<int64_t> dims(input.sizes.size());
    std::iota(dims.begin(), dims.end(), 0);
    std::swap(dims[static_cast<size_t>(d0)], dims[static_cast<size_t>(d1)]);
    Node synthetic = node;
    synthetic.inputs = {
        {.name = "self", .arg = require_arg(node, "self", 0).arg},
        {.name = "dims", .arg = IntListArg{dims, {}}},
    };
    lower_permute(synthetic);
  }

  void lower_copy_or_alias(const Node& node, bool alias) {
    const TensorRef input = tensor_arg(node, "self", 0);
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* in = input.tensor;
    ggml_tensor* result = nullptr;
    if (alias) {
      // ggml_view_tensor sets only view_src, so the graph would not reach the
      // producer of the aliased tensor.
      result = ggml_view_4d(
          ctx,
          in,
          in->ne[0],
          in->ne[1],
          in->ne[2],
          in->ne[3],
          in->nb[1],
          in->nb[2],
          in->nb[3],
          0);
    } else {
      result = materialize(in);
    }
    set_single_output(node, TensorRef{result, input.sizes});
  }

  void lower_cast(const Node& node) {
    // ggml_cpy reads any strides, so a deferred permutation is consumed here.
    const ValueId input_id = tensor_arg_id(node, "self", 0);
    const TensorRef input = view(input_id);
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error("ggml: cast must produce one output");
    }
    const TensorMeta& meta = method_.graph.value(ids.front()).tensor_meta();
    const ggml_type type = compute_type(meta);
    if (input.tensor->type == type) {
      // Already the target type: share it as materialize() would, keeping a
      // deferred view deferred.
      set_view(ids.front(), input);
      return;
    }
    // ggml-cpu converts I32 only to and from F32.
    ggml_tensor* source = input.tensor;
    if ((source->type == GGML_TYPE_I32) != (type == GGML_TYPE_I32) &&
        source->type != GGML_TYPE_F32 && type != GGML_TYPE_F32) {
      source = as_f32(source);
    }
    set(ids.front(),
        TensorRef{
            ggml_cast(instance_.context.get(), source, type), input.sizes});
  }

  ggml_tensor* as_f32(ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
        ? tensor
        : ggml_cast(instance_.context.get(), tensor, GGML_TYPE_F32);
  }

  ggml_tensor* as_i32(ggml_tensor* tensor) {
    if (tensor->type == GGML_TYPE_I32) {
      return tensor;
    }
    // Reuse conversions of an immutable arithmetic result (or one of its
    // views), e.g. an index consumed by several independent operations.
    // Never reuse a conversion of a graph input or a potentially writable
    // op; explicit in-place writes invalidate conversions of their storage.
    const ggml_tensor* root = storage_root(tensor);
    const bool reusable = tensor->type == GGML_TYPE_F32 &&
        (root->op == GGML_OP_ADD || root->op == GGML_OP_SUB ||
         root->op == GGML_OP_MUL || root->op == GGML_OP_DIV ||
         root->op == GGML_OP_ARANGE || root->op == GGML_OP_REPEAT);
    if (reusable) {
      if (const auto it = i32_casts_.find(tensor); it != i32_casts_.end()) {
        return it->second;
      }
    }
    ggml_tensor* result =
        ggml_cast(instance_.context.get(), tensor, GGML_TYPE_I32);
    if (reusable) {
      i32_casts_.emplace(tensor, result);
    }
    return result;
  }

  void invalidate_cached_transforms(const ggml_tensor* target) {
    const ggml_tensor* root = storage_root(target);
    for (auto it = i32_casts_.begin(); it != i32_casts_.end();) {
      if (storage_root(it->first) == root) {
        it = i32_casts_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = attention_masks_.begin(); it != attention_masks_.end();) {
      if (storage_root(tensor(it->first.first).tensor) == root) {
        it = attention_masks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  ggml_tensor* contiguous(ggml_tensor* tensor) {
    return ggml_is_contiguous(tensor)
        ? tensor
        : ggml_cont(instance_.context.get(), tensor);
  }

  // The values the graph writes in place, and those that share their storage
  // through view lowerings, so copies that touch them stay real copies.
  void index_written_values() {
    static const std::unordered_set<std::string> kViews = {
        "aten._unsafe_view.default",
        "aten.alias.default",
        "aten.permute.default",
        "aten.permute_copy.default",
        "aten.reshape.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.dims",
        "aten.squeeze_copy.dims",
        "aten.transpose_copy.int",
        "aten.unsqueeze.default",
        "aten.unsqueeze_copy.default",
        "aten.view.default",
        "aten.view_copy.default",
    };
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      const std::string key = node.is_call() ? op_key(node.target) : "";
      const NamedArgument* target = key == "aten.index_put_.default"
          ? find_arg(node, "self", 0)
          : key == "llama.update_cache.default" ? find_arg(node, "cache", 1)
                                                : nullptr;
      if (target != nullptr && target->arg.kind() == ArgKind::Tensor) {
        written_values_.insert(target->arg.as_tensor().id);
      }
    }
    for (bool grew = !written_values_.empty(); grew;) {
      grew = false;
      for (const NodeId id : method_.graph.schedule) {
        const Node& node = method_.graph.node(id);
        const NamedArgument* source = find_arg(node, "self", 0);
        if (!node.is_call() || !kViews.contains(op_key(node.target)) ||
            source == nullptr || source->arg.kind() != ArgKind::Tensor) {
          continue;
        }
        std::vector<ValueId> shared = output_ids(node);
        shared.push_back(source->arg.as_tensor().id);
        if (std::ranges::any_of(shared, [this](ValueId value) {
              return written_values_.contains(value);
            })) {
          for (const ValueId value : shared) {
            grew |= written_values_.insert(value).second;
          }
        }
      }
    }
  }

  bool touches_written_value(const Node& node) const {
    for (const NamedArgument& input : node.inputs) {
      if (input.arg.kind() == ArgKind::Tensor &&
          written_values_.contains(input.arg.as_tensor().id)) {
        return true;
      }
    }
    return std::ranges::any_of(output_ids(node), [this](ValueId value) {
      return written_values_.contains(value);
    });
  }

  // The result of a copying op. Contiguous storage is shared through a view
  // instead of copied unless it is persistent, already written, or the node
  // touches a value the graph writes in place (index_written_values).
  ggml_tensor* materialize(ggml_tensor* tensor) {
    const ggml_tensor* root = storage_root(tensor);
    if (copy_required_ || !ggml_is_contiguous(tensor) ||
        persistent_storage_.contains(root) || written_storage_.contains(root)) {
      return ggml_cont(instance_.context.get(), tensor);
    }
    aliased_storage_.insert(root);
    return ggml_reshape_4d(
        instance_.context.get(),
        tensor,
        tensor->ne[0],
        tensor->ne[1],
        tensor->ne[2],
        tensor->ne[3]);
  }

  void mark_written(const ggml_tensor* target) {
    const ggml_tensor* root = storage_root(target);
    if (aliased_storage_.contains(root)) {
      throw std::runtime_error(
          "ggml: in-place write to storage shared by a copy's view");
    }
    written_storage_.insert(root);
  }

  struct BinaryOperands {
    ggml_tensor* lhs = nullptr;
    ggml_tensor* rhs = nullptr;
    std::vector<int64_t> sizes;
  };

  BinaryOperands tensor_operands(
      const TensorRef& left,
      const TensorRef& right,
      bool force_f32) {
    BinaryOperands out{
        .lhs = left.tensor,
        .rhs = right.tensor,
        .sizes = broadcast_shape(left.sizes, right.sizes),
    };
    if (force_f32 || out.lhs->type != out.rhs->type) {
      out.lhs = as_f32(out.lhs);
      out.rhs = as_f32(out.rhs);
    }
    // ggml only broadcasts the right operand.
    if (reversed_extents(left.sizes) != reversed_extents(out.sizes)) {
      ggml_context* ctx = instance_.context.get();
      out.lhs =
          ggml_repeat(ctx, out.lhs, new_tensor(ctx, out.lhs->type, out.sizes));
    }
    return out;
  }

  void lower_binary(const Node& node, const std::string& key) {
    const TensorRef left = tensor_arg(node, "self", 0);
    const NamedArgument& rhs_arg = require_arg(node, "other", 1);
    if (float_arg(node, "alpha", 2, 1.0, true) != 1.0) {
      throw std::runtime_error("ggml: add/sub alpha must be one");
    }
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* result = nullptr;
    std::vector<int64_t> sizes = left.sizes;
    if (rhs_arg.arg.kind() == ArgKind::Tensor) {
      BinaryOperands operands = tensor_operands(
          left, tensor(rhs_arg.arg.as_tensor().id), /*force_f32=*/false);
      ggml_tensor* lhs = operands.lhs;
      ggml_tensor* right = operands.rhs;
      sizes = std::move(operands.sizes);
      if (key == "aten.add.Tensor") {
        result = ggml_add(ctx, lhs, right);
      } else if (key == "aten.mul.Tensor") {
        // Recognize the backend-independent arithmetic spellings of SiLU
        // and SwiGLU. A single ggml GLU node also lets CUDA use its generic
        // fused quantized matvec path when the operands qualify.
        const auto is_unary = [](const ggml_tensor* t, ggml_unary_op op) {
          return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
        };
        const auto matches_swiglu = [&](ggml_tensor* gate, ggml_tensor* up) {
          ggml_tensor* x = gate->src[0];
          return is_unary(gate, GGML_UNARY_OP_SILU) && x != nullptr &&
              gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
              ggml_are_same_shape(gate, up) && ggml_is_contiguous_1(x) &&
              ggml_is_contiguous_1(up);
        };
        const auto matches_silu = [&](ggml_tensor* sigmoid, ggml_tensor* x) {
          return is_unary(sigmoid, GGML_UNARY_OP_SIGMOID) &&
              sigmoid->src[0] == x && x->type == GGML_TYPE_F32 &&
              ggml_are_same_shape(sigmoid, x);
        };
        if (matches_swiglu(lhs, right)) {
          result = ggml_swiglu_split(ctx, lhs->src[0], right);
        } else if (matches_swiglu(right, lhs)) {
          result = ggml_swiglu_split(ctx, right->src[0], lhs);
        } else if (matches_silu(lhs, right)) {
          result = ggml_silu(ctx, right);
        } else if (matches_silu(right, lhs)) {
          result = ggml_silu(ctx, lhs);
        } else {
          result = ggml_mul(ctx, lhs, right);
        }
      } else if (key == "aten.sub.Tensor") {
        result = ggml_sub(ctx, lhs, right);
      } else {
        result = ggml_div(ctx, lhs, right);
      }
    } else {
      const float scalar = static_cast<float>(float_arg(node, "other", 1, 0));
      // ggml_scale only takes contiguous F32.
      ggml_tensor* lhs = contiguous(as_f32(left.tensor));
      if (key == "aten.mul.Tensor") {
        result = ggml_scale(ctx, lhs, scalar);
      } else if (key == "aten.div.Tensor") {
        result = ggml_scale(ctx, lhs, 1.0f / scalar);
      } else if (key == "aten.add.Tensor") {
        result = ggml_scale_bias(ctx, lhs, 1.0f, scalar);
      } else {
        result = ggml_scale_bias(ctx, lhs, 1.0f, -scalar);
      }
    }
    // Bools are carried as 0/1 floats, and Bool add is a logical or.
    const std::vector<ValueId> ids = output_ids(node);
    if (key == "aten.add.Tensor" && ids.size() == 1 &&
        method_.graph.value(ids.front()).tensor_meta().dtype ==
            ScalarType::Bool) {
      result = ggml_step(ctx, result);
    }
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  // ggml-cpu's unary kernels abort on integer tensors.
  static void require_float(const Node& node, const ggml_tensor* tensor) {
    if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 ||
        tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_I64) {
      throw std::runtime_error(
          "ggml: op '" + node.target +
          "' supports floating-point tensors only");
    }
  }

  using Unary = ggml_tensor* (*)(ggml_context*, ggml_tensor*);

  void lower_unary(const Node& node, Unary fn) {
    const TensorRef input = tensor_arg(node, "self", 0);
    require_float(node, input.tensor);
    set_single_output(
        node,
        TensorRef{fn(instance_.context.get(), input.tensor), input.sizes});
  }

  void lower_hardtanh(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    require_float(node, input.tensor);
    const float min =
        static_cast<float>(float_arg(node, "min_val", 1, -1.0, true));
    const float max =
        static_cast<float>(float_arg(node, "max_val", 2, 1.0, true));
    set_single_output(
        node,
        TensorRef{
            ggml_clamp(instance_.context.get(), input.tensor, min, max),
            input.sizes});
  }

  std::array<int, 2> pair_arg(
      const Node& node,
      std::string_view name,
      size_t positional,
      std::array<int, 2> fallback,
      int64_t min_value) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() == ArgKind::None) {
      return fallback;
    }
    const std::vector<int64_t> values = int_list(found->arg);
    if (values.empty()) {
      return fallback;
    }
    if (values.size() > 2) {
      throw std::runtime_error(
          "ggml: '" + std::string(name) + "' must have one or two entries");
    }
    for (const int64_t value : values) {
      if (value < min_value || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "ggml: '" + std::string(name) + "' entry is out of range");
      }
    }
    // PyTorch lists are (height, width); ggml takes width first.
    return {static_cast<int>(values.back()), static_cast<int>(values.front())};
  }

  std::optional<TensorRef> optional_tensor(
      const Node& node,
      std::string_view name,
      size_t positional) const {
    const NamedArgument* found = find_arg(node, name, positional);
    if (found == nullptr || found->arg.kind() != ArgKind::Tensor ||
        !valid(found->arg.as_tensor().id)) {
      return std::nullopt;
    }
    return tensor(found->arg.as_tensor().id);
  }

  // These lowerings produce only their first output, so a consumer of any
  // other output is rejected here instead of failing to resolve later.
  void require_only_first_output_used(const Node& node) const {
    const std::vector<ValueId> ids = output_ids(node);
    for (size_t i = 1; i < ids.size(); ++i) {
      if (!method_.graph.value(ids[i]).consumer_ids.empty() ||
          std::ranges::find(method_.graph.output_ids, ids[i]) !=
              method_.graph.output_ids.end()) {
        throw std::runtime_error(
            "ggml: op '" + node.target + "' lowers only its first output");
      }
    }
  }

  static void require_f32(const Node& node, const ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32) {
      throw std::runtime_error(
          "ggml: op '" + node.target + "' supports F32 tensors only");
    }
  }

  void lower_convolution(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const TensorRef weight = tensor_arg(node, "weight", 1);
    const size_t rank = input.sizes.size();
    if (bool_arg(node, "transposed", 6, false) || (rank != 3 && rank != 4) ||
        weight.sizes.size() != rank) {
      throw std::runtime_error(
          "ggml: convolution supports 1D and 2D non-transposed");
    }
    require_f32(node, input.tensor);
    require_f32(node, weight.tensor);
    auto stride = pair_arg(node, "stride", 3, {1, 1}, 1);
    auto padding = pair_arg(node, "padding", 4, {0, 0}, 0);
    auto dilation = pair_arg(node, "dilation", 5, {1, 1}, 1);
    const int64_t groups = int_arg(node, "groups", 8, 1, true);
    const int64_t channels = input.sizes[1];
    const int64_t out_channels = weight.sizes[0];
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* in = contiguous(input.tensor);
    ggml_tensor* kernel = contiguous(weight.tensor);
    if (rank == 3) {
      // [N, C, L] runs as a height-1 image.
      in = ggml_reshape_4d(
          ctx, in, input.sizes[2], 1, input.sizes[1], input.sizes[0]);
      kernel = ggml_reshape_4d(
          ctx, kernel, weight.sizes[2], 1, weight.sizes[1], weight.sizes[0]);
      stride[1] = 1;
      padding[1] = 0;
      dilation[1] = 1;
    }
    ggml_tensor* result = nullptr;
    if (groups == 1 && weight.sizes[1] == channels) {
      // ggml_conv_2d rounds the im2col columns of an F32 kernel through F16.
      ggml_tensor* columns = ggml_im2col(
          ctx,
          kernel,
          in,
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1],
          /*is_2D=*/true,
          GGML_TYPE_F32);
      result = ggml_mul_mat(
          ctx,
          ggml_reshape_2d(
              ctx,
              columns,
              columns->ne[0],
              columns->ne[1] * columns->ne[2] * columns->ne[3]),
          ggml_reshape_2d(ctx, kernel, columns->ne[0], out_channels));
      result = ggml_reshape_4d(
          ctx,
          result,
          columns->ne[1],
          columns->ne[2],
          columns->ne[3],
          out_channels);
      result = materialize(ggml_permute(ctx, result, 0, 1, 3, 2));
    } else if (
        groups == channels && out_channels == channels &&
        weight.sizes[1] == 1) {
      result = ggml_conv_2d_dw_direct(
          ctx,
          kernel,
          in,
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1]);
    } else {
      throw std::runtime_error(
          "ggml: convolution supports groups of 1 or depthwise only");
    }
    if (const std::optional<TensorRef> bias =
            optional_tensor(node, "bias", 2)) {
      if (bias->sizes != std::vector<int64_t>{out_channels}) {
        throw std::runtime_error(
            "ggml: convolution bias must hold one value per output channel");
      }
      result = ggml_add(
          ctx,
          result,
          ggml_reshape_4d(ctx, bias->tensor, 1, 1, out_channels, 1));
    }
    std::vector<int64_t> sizes{
        result->ne[3], result->ne[2], result->ne[1], result->ne[0]};
    if (rank == 3) {
      sizes.erase(sizes.begin() + 2);
      result = ggml_reshape_3d(ctx, result, sizes[2], sizes[1], sizes[0]);
    }
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  // Only the normalized output is lowered; the saved statistics are empty in
  // inference.
  void lower_batch_norm(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const std::optional<TensorRef> weight_ref =
        optional_tensor(node, "weight", 1);
    const std::optional<TensorRef> bias_ref = optional_tensor(node, "bias", 2);
    const TensorRef mean = tensor_arg(node, "running_mean", 3);
    const TensorRef var = tensor_arg(node, "running_var", 4);
    const double eps = float_arg(node, "eps", 6, 1e-5, true);
    if (input.sizes.size() < 2) {
      throw std::runtime_error("ggml: batch norm needs a channel dimension");
    }
    const std::vector<int64_t> per_channel{input.sizes[1]};
    if (mean.sizes != per_channel || var.sizes != per_channel ||
        (weight_ref && weight_ref->sizes != per_channel) ||
        (bias_ref && bias_ref->sizes != per_channel)) {
      throw std::runtime_error(
          "ggml: batch norm statistics must hold one value per channel");
    }
    require_only_first_output_used(node);
    ggml_tensor* weight = weight_ref ? weight_ref->tensor : nullptr;
    ggml_tensor* bias = bias_ref ? bias_ref->tensor : nullptr;
    ggml_context* ctx = instance_.context.get();
    // y = x * scale + shift with scale = weight / sqrt(var + eps) and
    // shift = bias - mean * scale.
    ggml_tensor* stddev = ggml_sqrt(
        ctx,
        ggml_scale_bias(
            ctx, as_f32(var.tensor), 1.0f, static_cast<float>(eps)));
    ggml_tensor* scale = ggml_div(
        ctx,
        weight != nullptr ? as_f32(weight)
                          : ggml_scale_bias(ctx, stddev, 0.0f, 1.0f),
        stddev);
    ggml_tensor* shift = ggml_mul(ctx, as_f32(mean.tensor), scale);
    shift = bias != nullptr ? ggml_sub(ctx, as_f32(bias), shift)
                            : ggml_neg(ctx, shift);
    std::vector<int64_t> channel_sizes(input.sizes.size(), 1);
    channel_sizes[1] = input.sizes[1];
    ggml_tensor* result =
        ggml_mul(ctx, input.tensor, reshape(ctx, scale, channel_sizes));
    result = ggml_add(ctx, result, reshape(ctx, shift, channel_sizes));
    set(output_ids(node).at(0), TensorRef{result, input.sizes});
  }

  // Indices are not produced; the lowering fails if they are consumed.
  void lower_max_pool2d(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    if (input.sizes.size() != 4) {
      throw std::runtime_error("ggml: max_pool2d needs an NCHW input");
    }
    const auto kernel = pair_arg(node, "kernel_size", 1, {0, 0}, 1);
    const auto stride = pair_arg(node, "stride", 2, kernel, 1);
    const auto padding = pair_arg(node, "padding", 3, {0, 0}, 0);
    const auto dilation = pair_arg(node, "dilation", 4, {1, 1}, 1);
    require_only_first_output_used(node);
    if (kernel[0] <= 0 || kernel[1] <= 0 ||
        dilation != std::array<int, 2>{1, 1} ||
        bool_arg(node, "ceil_mode", 5, false)) {
      throw std::runtime_error(
          "ggml: max_pool2d supports dilation 1 without ceil_mode");
    }
    require_f32(node, input.tensor);
    ggml_tensor* result = ggml_pool_2d(
        instance_.context.get(),
        contiguous(input.tensor),
        GGML_OP_POOL_MAX,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        static_cast<float>(padding[0]),
        static_cast<float>(padding[1]));
    std::vector<int64_t> sizes{
        result->ne[3], result->ne[2], result->ne[1], result->ne[0]};
    set(output_ids(node).at(0), TensorRef{result, std::move(sizes)});
  }

  // Reduces a run of trailing dimensions, which ggml_mean sees as one row.
  void lower_mean(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    require_f32(node, input.tensor);
    const NamedArgument* dtype = find_arg(node, "dtype", 3);
    if (dtype != nullptr && dtype->arg.kind() != ArgKind::None) {
      throw std::runtime_error("ggml: mean does not support dtype");
    }
    const size_t rank = input.sizes.size();
    const NamedArgument* dim_arg = find_arg(node, "dim", 1);
    std::vector<int64_t> dims;
    if (dim_arg != nullptr && dim_arg->arg.kind() != ArgKind::None) {
      dims = int_list(dim_arg->arg);
    }
    if (dims.empty()) {
      dims.resize(rank);
      std::iota(dims.begin(), dims.end(), 0);
    }
    for (int64_t& dim : dims) {
      dim = normalize_dim(dim, rank);
    }
    std::ranges::sort(dims);
    const size_t first = rank - dims.size();
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] != static_cast<int64_t>(first + i)) {
        throw std::runtime_error(
            "ggml: mean supports trailing dimensions only");
      }
    }
    const bool keepdim = bool_arg(node, "keepdim", 2, false);
    int64_t row = 1;
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < rank; ++i) {
      if (i < first) {
        sizes.push_back(input.sizes[i]);
      } else {
        row *= input.sizes[i];
        if (keepdim) {
          sizes.push_back(1);
        }
      }
    }
    if (row == 0) {
      throw std::runtime_error("ggml: mean over an empty dimension");
    }
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* rows = ggml_reshape_2d(
        ctx,
        contiguous(input.tensor),
        row,
        static_cast<int64_t>(checked_numel(input.sizes)) / row);
    ggml_tensor* result = reshape(ctx, ggml_mean(ctx, rows), sizes);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_slice(const Node& node, bool copy) {
    const TensorRef input = tensor_arg(node, "self", 0);
    const int64_t dim =
        normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
    int64_t start = int_arg(node, "start", 2, 0, true);
    int64_t end =
        int_arg(node, "end", 3, input.sizes[static_cast<size_t>(dim)], true);
    const int64_t step = int_arg(node, "step", 4, 1, true);
    if (step != 1) {
      throw std::runtime_error("ggml: slice step must be one");
    }
    const int64_t extent = input.sizes[static_cast<size_t>(dim)];
    start = std::clamp(start < 0 ? start + extent : start, int64_t{0}, extent);
    end = std::clamp(end < 0 ? end + extent : end, start, extent);
    std::vector<int64_t> sizes = input.sizes;
    sizes[static_cast<size_t>(dim)] = end - start;
    const size_t gdim = sizes.size() - 1 - static_cast<size_t>(dim);
    ggml_tensor* result = view_with_shape(
        instance_.context.get(),
        input.tensor,
        sizes,
        static_cast<size_t>(start) * input.tensor->nb[gdim]);
    if (copy) {
      result = materialize(result);
    }
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_select(const Node& node, bool copy) {
    const TensorRef input = tensor_arg(node, "self", 0);
    const int64_t dim =
        normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
    int64_t index = int_arg(node, "index", 2, 0);
    const int64_t extent = input.sizes[static_cast<size_t>(dim)];
    index = index < 0 ? index + extent : index;
    if (index < 0 || index >= extent) {
      throw std::runtime_error("ggml: select index is out of range");
    }
    std::vector<int64_t> view_sizes = input.sizes;
    view_sizes[static_cast<size_t>(dim)] = 1;
    const size_t gdim = view_sizes.size() - 1 - static_cast<size_t>(dim);
    ggml_tensor* result = view_with_shape(
        instance_.context.get(),
        input.tensor,
        view_sizes,
        static_cast<size_t>(index) * input.tensor->nb[gdim]);
    result = copy ? materialize(result) : contiguous(result);
    std::vector<int64_t> sizes = input.sizes;
    sizes.erase(sizes.begin() + dim);
    result = reshape(instance_.context.get(), result, sizes);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_unsqueeze(const Node& node) {
    TensorRef input = tensor_arg(node, "self", 0);
    if (op_key(node.target) == "aten.unsqueeze_copy.default") {
      input.tensor = snapshot(input.tensor);
    }
    int64_t dim = int_arg(node, "dim", 1, 0);
    const int64_t rank = static_cast<int64_t>(input.sizes.size() + 1);
    if (dim < 0) {
      dim += rank;
    }
    if (dim < 0 || dim >= rank) {
      throw std::runtime_error("ggml: unsqueeze dimension is out of range");
    }
    std::vector<int64_t> sizes = input.sizes;
    sizes.insert(sizes.begin() + dim, 1);
    set_single_output(
        node,
        TensorRef{
            reshape(instance_.context.get(), input.tensor, sizes), sizes});
  }

  void lower_squeeze(const Node& node) {
    TensorRef input = tensor_arg(node, "self", 0);
    if (op_key(node.target) == "aten.squeeze_copy.dims") {
      input.tensor = snapshot(input.tensor);
    }
    std::vector<int64_t> dims = int_list(require_arg(node, "dim", 1).arg);
    std::vector<bool> remove(input.sizes.size());
    for (int64_t dim : dims) {
      dim = normalize_dim(dim, input.sizes.size());
      if (input.sizes[static_cast<size_t>(dim)] == 1) {
        remove[static_cast<size_t>(dim)] = true;
      }
    }
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < input.sizes.size(); ++i) {
      if (!remove[i]) {
        sizes.push_back(input.sizes[i]);
      }
    }
    set_single_output(
        node,
        TensorRef{
            reshape(instance_.context.get(), input.tensor, sizes), sizes});
  }

  void lower_expand(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    std::vector<int64_t> sizes = int_list(require_arg(node, "size", 1).arg);
    if (sizes.size() < input.sizes.size()) {
      throw std::runtime_error("ggml: expand cannot drop dimensions");
    }
    // A mask expanded over its key prefix only (see key_extents_).
    if (const std::optional<int64_t> extent = key_extent_of(node)) {
      sizes.back() = *extent;
    }
    const size_t offset = sizes.size() - input.sizes.size();
    for (size_t i = 0; i < sizes.size(); ++i) {
      const int64_t in = i < offset ? 1 : input.sizes[i - offset];
      if (sizes[i] == -1 && i >= offset) {
        sizes[i] = in;
      }
      if (sizes[i] < 0 || (in != 1 && in != sizes[i])) {
        throw std::runtime_error(
            "ggml: expand size is incompatible with input");
      }
    }
    ggml_tensor* shape =
        new_tensor(instance_.context.get(), input.tensor->type, sizes);
    ggml_tensor* result =
        ggml_repeat(instance_.context.get(), input.tensor, shape);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_cat(const Node& node) {
    const Argument& tensors_arg = require_arg(node, "tensors", 0).arg;
    if (tensors_arg.kind() != ArgKind::TensorList) {
      throw std::runtime_error("ggml: cat expects a tensor list");
    }
    const std::vector<ValueId>& ids = tensors_arg.as_tensor_list().ids;
    if (ids.empty()) {
      throw std::runtime_error("ggml: cat tensor list is empty");
    }
    TensorRef result = tensor(ids.front());
    const int64_t dim =
        normalize_dim(int_arg(node, "dim", 1, 0, true), result.sizes.size());
    const int gdim = static_cast<int>(result.sizes.size() - 1 - dim);
    // Producers may carry one dtype as different ggml types (an F16 linear
    // result is F32), so mismatched operands take the output's type. ggml
    // converts I32 only to F32.
    const auto to_output_type = [&](ggml_tensor* t) {
      ggml_context* ctx = instance_.context.get();
      const ggml_type type = compute_type(
          method_.graph.value(output_ids(node).at(0)).tensor_meta());
      if (t->type == GGML_TYPE_I32 && type != GGML_TYPE_I32) {
        t = ggml_cast(ctx, t, GGML_TYPE_F32);
      }
      return t->type == type ? t : ggml_cast(ctx, t, type);
    };
    for (size_t i = 1; i < ids.size(); ++i) {
      TensorRef next = tensor(ids[i]);
      bool compatible = next.sizes.size() == result.sizes.size();
      for (size_t d = 0; compatible && d < next.sizes.size(); ++d) {
        compatible =
            d == static_cast<size_t>(dim) || next.sizes[d] == result.sizes[d];
      }
      if (!compatible) {
        throw std::runtime_error(
            "ggml: cat inputs must match in rank and every other dim");
      }
      if (next.tensor->type != result.tensor->type) {
        result.tensor = to_output_type(result.tensor);
        next.tensor = to_output_type(next.tensor);
      }
      result.tensor = ggml_concat(
          instance_.context.get(), result.tensor, next.tensor, gdim);
      result.sizes[static_cast<size_t>(dim)] +=
          next.sizes.at(static_cast<size_t>(dim));
    }
    set_single_output(node, std::move(result));
  }

  void lower_matmul(const Node& node) {
    const TensorRef left = tensor_arg(node, "self", 0);
    const TensorRef right = view_arg(node, "other", 1);
    if (left.sizes.size() < 2 || right.sizes.size() < 2) {
      throw std::runtime_error("ggml: matmul requires rank >= 2");
    }
    if (left.sizes.back() != right.sizes[right.sizes.size() - 2]) {
      throw std::runtime_error("ggml: matmul contracted dims differ");
    }
    // mul_mat broadcasts only its first operand, here the right batch.
    const size_t left_batch = left.sizes.size() - 2;
    const size_t right_batch = right.sizes.size() - 2;
    bool broadcastable = right_batch <= left_batch;
    for (size_t i = 1; broadcastable && i <= right_batch; ++i) {
      const int64_t dim = right.sizes[right_batch - i];
      broadcastable = dim == 1 || dim == left.sizes[left_batch - i];
    }
    if (!broadcastable) {
      throw std::runtime_error("ggml: matmul cannot broadcast the left batch");
    }
    ggml_context* ctx = instance_.context.get();
    // mul_mat(a, b) contracts ne[0] of both, so [M,K] @ [K,N] is
    // mul_mat(right^T, left). reshape() copies right^T only when it is strided
    // and also resets the strides of size-1 dims, which ggml_is_contiguous
    // ignores but mul_mat's transposed assert reads.
    std::vector<int64_t> right_t_sizes = right.sizes;
    std::swap(right_t_sizes.back(), right_t_sizes[right_batch]);
    ggml_tensor* right_t =
        reshape(ctx, ggml_transpose(ctx, right.tensor), right_t_sizes);
    ggml_tensor* result = ggml_mul_mat(ctx, right_t, left.tensor);
    std::vector<int64_t> sizes = left.sizes;
    sizes.back() = right.sizes.back();
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  void lower_update_cache(const Node& node) {
    const TensorRef update = tensor_arg(node, "value", 0);
    const TensorRef cache = tensor_arg(node, "cache", 1);
    const int64_t start = int_arg(node, "start_pos", 2, 0);
    if (cache.sizes.size() != update.sizes.size() || cache.sizes.size() != 4 ||
        update.sizes[0] != cache.sizes[0] ||
        update.sizes[2] != cache.sizes[2] ||
        update.sizes[3] != cache.sizes[3] || start < 0 ||
        start > cache.sizes[1] - update.sizes[1]) {
      throw std::runtime_error("ggml: invalid KV cache update");
    }
    std::vector<int64_t> sizes = cache.sizes;
    sizes[1] = update.sizes[1];
    const size_t offset = static_cast<size_t>(start) * cache.tensor->nb[2];
    ggml_tensor* destination =
        view_with_shape(instance_.context.get(), cache.tensor, sizes, offset);
    ggml_tensor* source = cache.tensor->type == GGML_TYPE_F16
        ? skip_f16_cast(update.tensor)
        : update.tensor;
    invalidate_cached_transforms(destination);
    mark_written(destination);
    ggml_tensor* write = ggml_cpy(instance_.context.get(), source, destination);
    expand_write(write);
    const std::vector<ValueId> ids = output_ids(node);
    if (!ids.empty()) {
      set(ids.front(), TensorRef{write, sizes});
    }
  }

  TensorRef active_cache(const TensorRef& cache, int64_t length) {
    if (cache.sizes.size() != 4 || length <= 0 || length > cache.sizes[1]) {
      throw std::runtime_error("ggml: invalid active KV-cache length");
    }
    std::vector<int64_t> sizes = cache.sizes;
    sizes[1] = length;
    return TensorRef{
        view_with_shape(instance_.context.get(), cache.tensor, sizes, 0),
        sizes};
  }

  TensorRef causal_mask(int64_t tokens, int64_t keys, int64_t start) {
    const std::vector<int64_t> sizes{1, 1, tokens, keys};
    ggml_tensor* mask =
        new_tensor(instance_.context.get(), GGML_TYPE_F16, sizes);
    ggml_set_input(mask);
    Upload upload;
    upload.tensor = mask;
    upload.bytes.resize(
        static_cast<size_t>(tokens * keys) * sizeof(ggml_fp16_t));
    auto* data = reinterpret_cast<ggml_fp16_t*>(upload.bytes.data());
    for (int64_t query = 0; query < tokens; ++query) {
      for (int64_t key = 0; key < keys; ++key) {
        const float value = key <= start + query
            ? 0.0f
            : -std::numeric_limits<float>::infinity();
        data[query * keys + key] = ggml_fp32_to_fp16(value);
      }
    }
    instance_.uploads.push_back(std::move(upload));
    return TensorRef{mask, sizes};
  }

  void lower_custom_sdpa(const Node& node) {
    const TensorRef query = tensor_arg(node, "query", 0);
    const TensorRef key_cache = tensor_arg(node, "key", 1);
    const TensorRef value_cache = tensor_arg(node, "value", 2);
    if (query.sizes.size() != 4) {
      throw std::runtime_error("ggml: custom_sdpa query must be [B,T,H,D]");
    }
    if (key_cache.sizes.size() != 4 || value_cache.sizes != key_cache.sizes ||
        key_cache.sizes[0] != query.sizes[0] ||
        key_cache.sizes[3] != query.sizes[3] || key_cache.sizes[2] <= 0 ||
        query.sizes[2] % key_cache.sizes[2] != 0) {
      throw std::runtime_error(
          "ggml: custom_sdpa q/k/v shapes are incompatible");
    }
    const int64_t start = int_arg(node, "start_pos", 3, 0);
    const int64_t tokens = query.sizes[1];
    if (start < 0 || start > key_cache.sizes[1] - tokens) {
      throw std::runtime_error(
          "ggml: custom_sdpa start_pos is outside the KV cache");
    }
    const NamedArgument* mask_arg = find_arg(node, "attn_mask", 4);
    const bool masked = mask_arg != nullptr &&
        mask_arg->arg.kind() == ArgKind::Tensor &&
        valid(mask_arg->arg.as_tensor().id);
    // As in ExecuTorch's kernel, a masked call ignores start_pos and attends
    // the whole cache.
    const int64_t keys = masked ? key_cache.sizes[1] : start + tokens;
    const bool is_causal = bool_arg(node, "is_causal", 6, false);
    if (float_arg(node, "dropout_p", 5, 0.0, true) != 0.0) {
      throw std::runtime_error("ggml: custom_sdpa dropout is unsupported");
    }
    const TensorRef key = active_cache(key_cache, keys);
    const TensorRef val = active_cache(value_cache, keys);
    // flash_attn_ext kernels read the query as F32.
    ggml_tensor* q = query.tensor->type == GGML_TYPE_F32
        ? query.tensor
        : ggml_cast(instance_.context.get(), query.tensor, GGML_TYPE_F32);
    q = ggml_permute(instance_.context.get(), q, 0, 2, 1, 3);
    ggml_tensor* k =
        ggml_permute(instance_.context.get(), key.tensor, 0, 2, 1, 3);
    ggml_tensor* v =
        ggml_permute(instance_.context.get(), val.tensor, 0, 2, 1, 3);

    ggml_tensor* mask = nullptr;
    if (masked) {
      if (is_causal) {
        throw std::runtime_error(
            "ggml: custom_sdpa cannot take a mask and is_causal");
      }
      const TensorRef attn_mask = tensor(mask_arg->arg.as_tensor().id);
      const size_t rank = attn_mask.sizes.size();
      // flash_attn_ext reads [tokens, keys] mask rows without bounds checks.
      if (rank < 2 || attn_mask.sizes[rank - 2] < tokens ||
          attn_mask.sizes[rank - 1] < keys) {
        throw std::runtime_error(
            "ggml: custom_sdpa mask does not cover every query and key");
      }
      mask =
          ggml_cast(instance_.context.get(), attn_mask.tensor, GGML_TYPE_F16);
    } else if (is_causal && tokens > 1) {
      mask = causal_mask(tokens, keys, start).tensor;
    }
    const double default_scale =
        1.0 / std::sqrt(static_cast<double>(query.sizes.back()));
    const float scale =
        static_cast<float>(float_arg(node, "scale", 7, default_scale, true));
    ggml_tensor* result = ggml_flash_attn_ext(
        instance_.context.get(), q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    set_single_output(node, TensorRef{result, query.sizes});
  }

  // Comparisons produce 0/1 floats: a <= b is 1 - step(a - b) and a < b is
  // step(b - a).
  void lower_compare(const Node& node, bool strict) {
    TensorRef left = tensor_arg(node, "self", 0);
    const NamedArgument& other = require_arg(node, "other", 1);
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* difference = nullptr;
    std::vector<int64_t> sizes = left.sizes;
    if (other.arg.kind() == ArgKind::Tensor) {
      TensorRef right = tensor(other.arg.as_tensor().id);
      if (const std::optional<int64_t> extent = key_extent_of(node)) {
        // An attention mask over a key prefix: only the per-key operand
        // spans the key axis.
        left = last_axis_prefix(left, *extent);
        right = last_axis_prefix(right, *extent);
      }
      BinaryOperands operands =
          tensor_operands(left, right, /*force_f32=*/true);
      difference = ggml_sub(ctx, operands.lhs, operands.rhs);
      sizes = std::move(operands.sizes);
    } else {
      difference = ggml_scale_bias(
          ctx,
          contiguous(as_f32(left.tensor)),
          1.0f,
          -static_cast<float>(float_arg(node, "other", 1, 0)));
    }
    ggml_tensor* result = strict
        ? ggml_step(ctx, ggml_neg(ctx, difference))
        : ggml_scale_bias(ctx, ggml_step(ctx, difference), -1.0f, 1.0f);
    set_single_output(node, TensorRef{result, std::move(sizes)});
  }

  // The result is F32, which holds integers exactly up to 2^24, and ggml
  // derives the length from the F32 bounds.
  void lower_arange(const Node& node) {
    const double start = float_arg(node, "start", 0, 0.0);
    const double end = float_arg(node, "end", 1, 0.0);
    const double step = float_arg(node, "step", 2, 1.0, true);
    if (step <= 0.0 || end <= start) {
      throw std::runtime_error(
          "ggml: arange needs a positive step and end above start");
    }
    constexpr double kExactF32 = 1 << 24;
    if (std::abs(start) > kExactF32 || std::abs(end) > kExactF32) {
      throw std::runtime_error("ggml: arange bounds must be within 2^24");
    }
    ggml_tensor* result = ggml_arange(
        instance_.context.get(),
        static_cast<float>(start),
        static_cast<float>(end),
        static_cast<float>(step));
    if (result->ne[0] !=
        static_cast<int64_t>(std::ceil((end - start) / step))) {
      throw std::runtime_error("ggml: arange length changes in F32");
    }
    set_single_output(node, TensorRef{result, {result->ne[0]}});
  }

  DynamicQuant dynamic_quant_arg(
      const Node& node,
      std::string_view name,
      size_t positional) const {
    const Argument& arg = require_arg(node, name, positional).arg;
    const auto* quant = arg.kind() == ArgKind::Tensor
        ? std::get_if<DynamicQuant>(&value(arg.as_tensor().id))
        : nullptr;
    if (quant == nullptr) {
      throw std::runtime_error(
          "ggml: '" + node.target +
          "' is only supported inside a dynamic activation quant chain");
    }
    return *quant;
  }

  void set_dynamic_quant(const Node& node, const DynamicQuant& quant) {
    for (const ValueId id : output_ids(node)) {
      set(id, quant);
    }
  }

  // ggml backends quantize the activations of a quantized matmul themselves,
  // in their own block format, so torchao's per-token int8 round trip is
  // dropped rather than emulated.
  void lower_choose_qparams(const Node& node) {
    const TensorRef input = tensor_arg(node, "input", 0);
    const Argument& target_dtype = require_arg(node, "target_dtype", 3).arg;
    if (target_dtype.kind() != ArgKind::ScalarType ||
        target_dtype.as_scalar_type().value != ScalarType::Char) {
      throw std::runtime_error(
          "ggml: dynamic activation quantization must target int8");
    }
    std::vector<int64_t> per_token(input.sizes.size(), 1);
    per_token.back() = input.sizes.back();
    if (int_list(require_arg(node, "block_size", 2).arg) != per_token) {
      throw std::runtime_error(
          "ggml: only per-token dynamic activation quantization is supported");
    }
    set_dynamic_quant(node, DynamicQuant{input});
  }

  void lower_quantize_affine(const Node& node) {
    const DynamicQuant quant = dynamic_quant_arg(node, "scale", 2);
    if (tensor_arg(node, "input", 0).tensor != quant.input.tensor) {
      throw std::runtime_error(
          "ggml: quantize_affine input differs from its qparams source");
    }
    set_dynamic_quant(node, quant);
  }

  void lower_dequantize_affine(const Node& node) {
    const DynamicQuant quant = dynamic_quant_arg(node, "input", 0);
    if (dynamic_quant_arg(node, "scale", 2).input.tensor !=
        quant.input.tensor) {
      throw std::runtime_error(
          "ggml: dequantize_affine scale differs from its input chain");
    }
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error(
          "ggml: dequantize_affine must produce one tensor");
    }
    if (std::ranges::find(method_.graph.output_ids, ids.front()) !=
        method_.graph.output_ids.end()) {
      throw std::runtime_error(
          "ggml: dynamically quantized activations must not be graph outputs");
    }
    for (const NodeId consumer :
         method_.graph.value(ids.front()).consumer_ids) {
      const Node& linear = method_.graph.node(consumer);
      const NamedArgument* input = find_arg(linear, "input", 0);
      if (op_key(linear.target) != "aten.linear.default" || input == nullptr ||
          input->arg.kind() != ArgKind::Tensor ||
          input->arg.as_tensor().id != ids.front() ||
          !ggml_is_quantized(tensor_arg(linear, "weight", 1).tensor->type)) {
        throw std::runtime_error(
            "ggml: dynamically quantized activations must feed quantized "
            "linear inputs");
      }
    }
    set(ids.front(), quant.input);
  }

  ggml_tensor* rope_factor_tensor(ValueId inv_freq) {
    if (const auto cached = rope_factor_tensors_.find(inv_freq);
        cached != rope_factor_tensors_.end()) {
      return cached->second;
    }
    const auto found = rope_factor_bytes_.find(inv_freq);
    if (found == rope_factor_bytes_.end()) {
      throw std::runtime_error("ggml: rope inv_freq was not indexed at load");
    }
    ggml_tensor* factors = ggml_new_tensor_1d(
        instance_.context.get(),
        GGML_TYPE_F32,
        static_cast<int64_t>(found->second.size() / sizeof(float)));
    ggml_set_input(factors);
    instance_.uploads.push_back(Upload{factors, found->second});
    rope_factor_tensors_.emplace(inv_freq, factors);
    return factors;
  }

  void lower_rope(const Node& node) {
    // ggml_rope_ext reads strided rows; rotating the [D, heads, tokens] view
    // of a transposed projection then needs no copy.
    const TensorRef input = rows_arg(node, "input", 0);
    const TensorRef positions = tensor_arg(node, "position_ids", 1);
    ggml_tensor* factors =
        rope_factor_tensor(require_arg(node, "inv_freq", 2).arg.as_tensor().id);
    const bool interleaved = bool_arg(node, "interleaved", 3, false);
    const double scale = float_arg(node, "attention_scale", 4, 1.0, true);
    const int n_dims = 2 * static_cast<int>(ggml_nelements(factors));
    if (input.sizes.size() != 4 || positions.sizes.size() != 2 ||
        positions.sizes[0] != 1 || positions.sizes[1] != input.sizes[2] ||
        n_dims > input.sizes[3]) {
      throw std::runtime_error(
          "ggml: rope expects input [B,H,T,D] and position_ids [1,T]");
    }
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* pos = ggml_reshape_1d(
        ctx, contiguous(as_i32(positions.tensor)), positions.sizes[1]);
    // ggml_rope_ext rotates [D, heads, tokens, B]; torch order is [B,H,T,D].
    ggml_tensor* rotated = ggml_rope_ext(
        ctx,
        ggml_permute(ctx, input.tensor, 0, 2, 1, 3),
        pos,
        factors,
        n_dims,
        interleaved ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX,
        /*n_ctx_orig=*/0,
        /*freq_base=*/1.0f,
        /*freq_scale=*/1.0f,
        /*ext_factor=*/0.0f,
        static_cast<float>(scale),
        /*beta_fast=*/0.0f,
        /*beta_slow=*/0.0f);
    set_single_view(
        node, TensorRef{ggml_permute(ctx, rotated, 0, 2, 1, 3), input.sizes});
  }

  // set_rows into an F16 destination and attention over F16 keys round F32
  // values exactly like a cast, so they can read a fresh F32 -> F16 cast's
  // source directly. ggml_cast is a CPY whose destination is its own result.
  // Skipping the cast moves the read of its source to the consumer, so a
  // source in a persistent buffer or in storage the graph writes in place
  // keeps its cast.
  ggml_tensor* skip_f16_cast(ggml_tensor* rows) const {
    ggml_tensor* source = rows->src[0];
    if (rows->type != GGML_TYPE_F16 || rows->op != GGML_OP_CPY ||
        rows->src[1] != rows || source->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(rows, source) ||
        !ggml_is_contiguous_rows(source)) {
      return rows;
    }
    const ggml_tensor* root = storage_root(source);
    return persistent_storage_.contains(root) || written_storage_.contains(root)
        ? rows
        : source;
  }

  void lower_index_put(const Node& node) {
    const TensorRef self = tensor_arg(node, "self", 0);
    const Argument& indices_arg = require_arg(node, "indices", 1).arg;
    // set_rows reads rows with any outer strides, so the values may be a
    // deferred permutation.
    const TensorRef values = rows_arg(node, "values", 2);
    const size_t rank = self.sizes.size();
    const std::vector<ValueId> indices =
        indices_arg.kind() == ArgKind::OptionalTensorList
        ? indices_arg.as_optional_tensor_list().ids
        : std::vector<ValueId>{};
    const auto is_index = [](ValueId id) { return valid(id); };
    const bool rows_only = rank >= 2 && indices.size() == rank - 1 &&
        is_index(indices.back()) &&
        std::none_of(indices.begin(), indices.end() - 1, is_index);
    const TensorRef index = rows_only ? tensor(indices.back()) : TensorRef{};
    std::vector<int64_t> row_sizes = self.sizes;
    if (rows_only && index.sizes.size() == 1) {
      row_sizes[rank - 2] = index.sizes[0];
    }
    if (!rows_only || index.sizes.size() != 1 || values.sizes != row_sizes ||
        bool_arg(node, "accumulate", 3, false)) {
      throw std::runtime_error(
          "ggml: index_put_ only supports writing whole rows along dim -2");
    }
    const ScalarType index_dtype =
        method_.graph.value(indices.back()).tensor_meta().dtype;
    if (index_dtype != ScalarType::Long && index_dtype != ScalarType::Int) {
      throw std::runtime_error(
          "ggml: index_put_ indices must be int64 or int32");
    }
    check_row_indices(indices.back(), self.sizes[rank - 2]);
    ggml_tensor* rows = values.tensor->type == GGML_TYPE_F16
        ? values.tensor
        : as_f32(values.tensor);
    if (self.tensor->type == GGML_TYPE_F16) {
      rows = skip_f16_cast(rows);
    }
    invalidate_cached_transforms(self.tensor);
    mark_written(self.tensor);
    ggml_tensor* result = ggml_set_rows(
        instance_.context.get(), self.tensor, rows, as_i32(index.tensor));
    expand_write(result);
    set_single_output(node, TensorRef{result, self.sizes});
  }

  ggml_tensor* attention_mask(ValueId id, int64_t tokens, int64_t keys) {
    const TensorRef mask = tensor(id);
    const size_t rank = mask.sizes.size();
    if (rank < 2 || rank > 4 ||
        (mask.sizes[rank - 2] != tokens && mask.sizes[rank - 2] != 1) ||
        mask.sizes.back() != keys) {
      throw std::runtime_error("ggml: attention mask must end in [T|1, S]");
    }
    // Out-of-place repeat/scale results, and views of them, are stable until
    // an explicit graph write to their storage (which invalidates this cache).
    // Reuse their additive F16 mask across attention ops; never cache
    // writable inputs.
    const ggml_tensor* root = storage_root(mask.tensor);
    const bool reusable =
        root->op == GGML_OP_REPEAT || root->op == GGML_OP_SCALE;
    const auto cache_key = std::make_pair(id, tokens);
    if (reusable) {
      const auto found = attention_masks_.find(cache_key);
      if (found != attention_masks_.end()) {
        return found->second;
      }
    }
    ggml_context* ctx = instance_.context.get();
    ggml_tensor* additive = mask.tensor;
    if (method_.graph.value(id).tensor_meta().dtype == ScalarType::Bool) {
      // max - max is exactly 0; -max saturates to -inf in F16.
      constexpr float kMax = std::numeric_limits<float>::max();
      additive =
          ggml_scale_bias(ctx, contiguous(as_f32(mask.tensor)), kMax, -kMax);
    }
    // flash_attn_ext broadcasts the mask over heads and batch but reads one
    // row per query.
    if (mask.sizes[rank - 2] != tokens) {
      std::vector<int64_t> rows = mask.sizes;
      rows[rank - 2] = tokens;
      additive = ggml_repeat(
          ctx, as_f32(additive), new_tensor(ctx, GGML_TYPE_F32, rows));
    }
    // flash_attn_ext needs a contiguous F16 mask.
    ggml_tensor* result = additive->type == GGML_TYPE_F16
        ? contiguous(additive)
        : ggml_cast(ctx, additive, GGML_TYPE_F16);
    if (reusable) {
      attention_masks_.emplace(cache_key, result);
    }
    return result;
  }

  void lower_sdpa(const Node& node) {
    // flash_attn_ext takes strided q/k/v with contiguous rows, e.g. the
    // head-transposed projections, without copies.
    const TensorRef query = rows_arg(node, "query", 0);
    const TensorRef key = rows_arg(node, "key", 1);
    const TensorRef val = rows_arg(node, "value", 2);
    const bool is_causal = bool_arg(node, "is_causal", 5, false);
    if (query.sizes.size() != 4 || key.sizes.size() != 4 ||
        val.sizes.size() != 4 ||
        float_arg(node, "dropout_p", 4, 0.0, true) != 0.0) {
      throw std::runtime_error(
          "ggml: scaled_dot_product_attention expects [B,H,T,D] without dropout");
    }
    if (key.sizes[0] != query.sizes[0] || val.sizes[0] != query.sizes[0] ||
        key.sizes[1] != val.sizes[1] || key.sizes[1] <= 0 ||
        query.sizes[1] % key.sizes[1] != 0 || key.sizes[2] != val.sizes[2] ||
        key.sizes[3] != query.sizes[3]) {
      throw std::runtime_error(
          "ggml: scaled_dot_product_attention q/k/v shapes are incompatible");
    }
    const int64_t tokens = query.sizes[2];
    int64_t keys = key.sizes[2];
    ggml_tensor* k = key.tensor;
    ggml_tensor* v = val.tensor;
    ggml_tensor* mask = nullptr;
    const NamedArgument* mask_arg = find_arg(node, "attn_mask", 3);
    if (mask_arg != nullptr && mask_arg->arg.kind() == ArgKind::Tensor &&
        valid(mask_arg->arg.as_tensor().id)) {
      if (is_causal) {
        throw std::runtime_error("ggml: sdpa cannot take a mask and is_causal");
      }
      const ValueId mask_id = mask_arg->arg.as_tensor().id;
      if (const auto extent = key_extents_.find(mask_id);
          extent != key_extents_.end()) {
        // Every query masks the keys past the extent: attend the prefix.
        keys = extent->second;
        const auto prefix = [&](ggml_tensor* t) {
          return ggml_view_4d(
              instance_.context.get(),
              t,
              t->ne[0],
              keys,
              t->ne[2],
              t->ne[3],
              t->nb[1],
              t->nb[2],
              t->nb[3],
              0);
        };
        k = prefix(k);
        v = prefix(v);
      }
      mask = attention_mask(mask_id, tokens, keys);
    } else if (is_causal) {
      mask = causal_mask(tokens, keys, 0).tensor;
    }
    const double default_scale =
        1.0 / std::sqrt(static_cast<double>(query.sizes.back()));
    const float scale =
        static_cast<float>(float_arg(node, "scale", 6, default_scale, true));
    ggml_context* ctx = instance_.context.get();
    // The kernels read the query as F32 and round it to the F16 key's dot
    // product type themselves (CPU vec_dot, Metal and CUDA shared memory), so
    // an F32 -> F16 cast that only makes the query match an F16 cache for
    // torch's SDPA is the identity on the result. Use its F32 source, where
    // skip_f16_cast allows, rather than round trip through F16.
    ggml_tensor* q = query.tensor;
    if (key.tensor->type == GGML_TYPE_F16) {
      q = skip_f16_cast(q);
    }
    ggml_tensor* result =
        ggml_flash_attn_ext(ctx, as_f32(q), k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    // flash_attn_ext returns [Dv, H, T, B]; torch order is [B, H, T, Dv].
    set_single_view(
        node,
        TensorRef{
            ggml_permute(ctx, result, 0, 2, 1, 3),
            {query.sizes[0], query.sizes[1], tokens, val.sizes[3]}});
  }

  void lower_sym_size(const Node& node) {
    const TensorRef input = tensor_arg(node, "self", 0);
    const int64_t dim =
        normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error("ggml: sym_size must produce one value");
    }
    set(ids.front(), input.sizes[static_cast<size_t>(dim)]);
  }

  int64_t symbolic_operand(const Argument& arg) const {
    if (arg.kind() == ArgKind::Int) {
      return int_value(arg);
    }
    throw std::runtime_error("ggml: symbolic arithmetic expects integers");
  }

  void lower_symbolic(const Node& node, const std::string& key) {
    if (node.inputs.size() < 2) {
      throw std::runtime_error("ggml: symbolic arithmetic is missing operands");
    }
    const int64_t a = symbolic_operand(node.inputs[0].arg);
    const int64_t b = symbolic_operand(node.inputs[1].arg);
    int64_t result = 0;
    bool overflow = false;
    if (key == "_operator.add") {
      overflow = __builtin_add_overflow(a, b, &result);
    } else if (key == "_operator.sub") {
      overflow = __builtin_sub_overflow(a, b, &result);
    } else if (key == "_operator.mul") {
      overflow = __builtin_mul_overflow(a, b, &result);
    } else {
      if (b == 0) {
        throw std::runtime_error("ggml: symbolic floor division by zero");
      }
      overflow = a == std::numeric_limits<int64_t>::min() && b == -1;
      result = overflow ? 0 : a / b;
      if (!overflow && a % b != 0 && (a < 0) != (b < 0)) {
        --result;
      }
    }
    if (overflow) {
      throw std::runtime_error("ggml: symbolic integer arithmetic overflows");
    }
    const std::vector<ValueId> ids = output_ids(node);
    if (ids.size() != 1) {
      throw std::runtime_error("ggml: symbolic op must produce one value");
    }
    set(ids.front(), result);
  }

  const Method& method_;
  const std::vector<InputSlot>& inputs_;
  const std::vector<OutputSlot>& outputs_;
  const std::unordered_map<ValueId, ggml_tensor*>& persistent_;
  const std::unordered_map<ValueId, std::vector<uint8_t>>& rope_factor_bytes_;
  // Mask values (and their producers) computed over a key prefix only.
  std::unordered_map<ValueId, int64_t> key_extents_;
  size_t graph_size_;
  std::vector<RuntimeValue> values_;
  std::unordered_map<ValueId, ggml_tensor*> rope_factor_tensors_;
  std::map<std::pair<ValueId, int64_t>, ggml_tensor*> attention_masks_;
  std::unordered_map<ggml_tensor*, ggml_tensor*> i32_casts_;
  std::unordered_set<const ggml_tensor*> persistent_storage_;
  std::unordered_set<const ggml_tensor*> aliased_storage_;
  std::unordered_set<const ggml_tensor*> written_storage_;
  std::unordered_set<ValueId> written_values_;
  // Set while lowering a node that touches a written value.
  bool copy_required_ = false;
  // Values lowered as non-contiguous views of shared storage, and their
  // copies.
  std::unordered_set<ValueId> deferred_;
  mutable std::unordered_map<ValueId, ggml_tensor*> deferred_copies_;
  GraphInstance instance_;
};

GraphInstance GgmlExecutable::build_graph(
    const std::vector<int64_t>& key_extents) {
  std::unordered_map<ValueId, int64_t> truncated;
  for (size_t i = 0; i < mask_extents_.size(); ++i) {
    if (key_extents.at(i) < mask_extents_[i].keys) {
      for (const ValueId id : mask_extents_[i].chain) {
        truncated.emplace(id, key_extents[i]);
      }
    }
  }
  GraphInstance instance = GraphBuilder(
                               method_,
                               inputs_,
                               outputs_,
                               persistent_,
                               rope_freq_factors_,
                               std::move(truncated),
                               options_.graph_size)
                               .build();
  if (options_.print_graph_ops) {
    std::map<std::string, int> counts;
    for (int i = 0; i < ggml_graph_n_nodes(instance.graph); ++i) {
      ++counts[ggml_op_desc(ggml_graph_node(instance.graph, i))];
    }
    std::string line = "ggml graph ops:";
    for (const auto& [op, count] : counts) {
      line += " " + op + "=" + std::to_string(count);
    }
    std::fprintf(stderr, "%s\n", line.c_str());
  }
  return instance;
}

} // namespace

ggml_backend_dev_t select_ggml_gpu_device(const GgmlEngineOptions& options) {
  int remaining = options.device_index;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
      continue;
    }
    if (remaining-- == 0) {
      return device;
    }
  }
  throw std::runtime_error(
      "ggml: GPU device " + std::to_string(options.device_index) +
      " is unavailable (" + std::to_string(options.device_index - remaining) +
      " GPU devices registered by the linked ggml backends)");
}

std::unique_ptr<EngineExecutable> make_ggml_executable(
    const Method& method,
    const Package& package,
    const GgmlEngineOptions& options,
    size_t* materialized_constant_bytes,
    size_t* heap_copied_constant_bytes,
    GgmlMutableKeyOwners& mutable_key_owners) {
  return std::make_unique<GgmlExecutable>(
      method,
      package,
      options,
      materialized_constant_bytes,
      heap_copied_constant_bytes,
      mutable_key_owners);
}

} // namespace ptn
