// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The deserializer bridge: native_backend::Program (FlatBuffer) -> ptn
// in-memory IR (Method / Graph / Node / Argument / Value). In-graph references
// (SSA names) are resolved to list ValueIds; per-graph namespaces are
// resolved independently (each HOP subgraph rebuilds its own name -> id map).
//
// Everything here runs on a buffer Program::load() has already put through
// flatbuffers::Verifier, so accessors return non-null wherever the schema
// declares the field required and wherever a union discriminator matches; the
// walkers below dereference those results directly. Fields the schema leaves
// optional are still checked, because verification says nothing about whether
// they are present. Vector<Offset<T>>::Get() computes an address rather than a
// nullable pointer, and verification bounds-checks every referenced object.

#include <executorch/backends/native/runtime/Program.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {
namespace {

std::string str_of(const flatbuffers::String* s) {
  return s != nullptr ? s->str() : std::string();
}

bool nonempty(const flatbuffers::String* s) {
  return s != nullptr && s->size() > 0;
}

ScalarType map_scalar_type(fbs::ScalarType t) {
  // ptn::ScalarType ids are pinned to the schema's, so the byte maps straight.
  const auto result = static_cast<ScalarType>(static_cast<int8_t>(t));
  element_size(result);
  return result;
}

QuantRoundingMode map_quant_rounding_mode(fbs::QuantRoundingMode mode) {
  switch (mode) {
    case fbs::QuantRoundingMode::TO_NEAREST_EVEN:
      return QuantRoundingMode::ToNearestEven;
    case fbs::QuantRoundingMode::AWAY_FROM_ZERO:
      return QuantRoundingMode::AwayFromZero;
    case fbs::QuantRoundingMode::TOWARD_ZERO:
      return QuantRoundingMode::TowardZero;
    case fbs::QuantRoundingMode::FLOOR:
      return QuantRoundingMode::Floor;
    case fbs::QuantRoundingMode::CEIL:
      return QuantRoundingMode::Ceil;
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantRoundingMode value " +
          std::to_string(static_cast<int>(mode)));
  }
}

QuantBitOrder map_quant_bit_order(fbs::QuantBitOrder order) {
  switch (order) {
    case fbs::QuantBitOrder::LSB_FIRST:
      return QuantBitOrder::LsbFirst;
    case fbs::QuantBitOrder::MSB_FIRST:
      return QuantBitOrder::MsbFirst;
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantBitOrder value " +
          std::to_string(static_cast<int>(order)));
  }
}

QuantSignedEncoding map_quant_signed_encoding(
    fbs::QuantSignedEncoding encoding) {
  switch (encoding) {
    case fbs::QuantSignedEncoding::UNSIGNED:
      return QuantSignedEncoding::Unsigned;
    case fbs::QuantSignedEncoding::TWOS_COMPLEMENT:
      return QuantSignedEncoding::TwosComplement;
    case fbs::QuantSignedEncoding::OFFSET:
      return QuantSignedEncoding::Offset;
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantSignedEncoding value " +
          std::to_string(static_cast<int>(encoding)));
  }
}

bool is_float_type(ScalarType dtype) {
  return dtype == ScalarType::Half || dtype == ScalarType::Float ||
      dtype == ScalarType::Double || dtype == ScalarType::BFloat16;
}

bool is_integer_type(ScalarType dtype) {
  return dtype == ScalarType::Byte || dtype == ScalarType::Char ||
      dtype == ScalarType::Short || dtype == ScalarType::Int ||
      dtype == ScalarType::Long || dtype == ScalarType::UInt16 ||
      dtype == ScalarType::UInt32 || dtype == ScalarType::UInt64;
}

QuantParam build_quant_param(const fbs::QuantParam* param) {
  if (param == nullptr || param->value() == nullptr) {
    throw std::runtime_error("build_tensor_meta: missing quant parameter");
  }
  switch (param->value_type()) {
    case fbs::QuantParamValue::InlineFloatQuantParam: {
      const auto* value = param->value_as_InlineFloatQuantParam();
      InlineFloatQuantParam out;
      out.value = value->value();
      out.dtype = map_scalar_type(value->dtype());
      if (!is_float_type(out.dtype) || !std::isfinite(out.value)) {
        throw std::runtime_error(
            "build_tensor_meta: invalid inline floating quant parameter");
      }
      return out;
    }
    case fbs::QuantParamValue::InlineIntQuantParam: {
      const auto* value = param->value_as_InlineIntQuantParam();
      InlineIntQuantParam out;
      out.value = value->value();
      out.dtype = map_scalar_type(value->dtype());
      if (!is_integer_type(out.dtype)) {
        throw std::runtime_error(
            "build_tensor_meta: invalid inline integer quant parameter");
      }
      return out;
    }
    case fbs::QuantParamValue::ExternalQuantParam: {
      const auto* value = param->value_as_ExternalQuantParam();
      ExternalQuantParam out;
      out.data_key = str_of(value->data_key());
      out.dtype = map_scalar_type(value->dtype());
      if (out.data_key.empty()) {
        throw std::runtime_error(
            "build_tensor_meta: external quant parameter has an empty data key");
      }
      return out;
    }
    case fbs::QuantParamValue::NONE:
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantParamValue value " +
          std::to_string(static_cast<int>(param->value_type())));
  }
}

QuantizedStorage build_quantized_storage(const fbs::QuantizedStorage* storage) {
  if (storage == nullptr || storage->value() == nullptr) {
    throw std::runtime_error("build_tensor_meta: missing quantized storage");
  }
  switch (storage->value_type()) {
    case fbs::QuantizedStorageValue::DenseQuantizedStorage:
      return DenseQuantizedStorage{};
    case fbs::QuantizedStorageValue::PackedBitsQuantizedStorage: {
      const auto* value = storage->value_as_PackedBitsQuantizedStorage();
      PackedBitsQuantizedStorage out;
      out.bit_width = value->bit_width();
      out.bit_order = map_quant_bit_order(value->bit_order());
      out.signed_encoding = map_quant_signed_encoding(value->signed_encoding());
      out.storage_offset = value->storage_offset();
      return out;
    }
    case fbs::QuantizedStorageValue::NONE:
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantizedStorageValue value " +
          std::to_string(static_cast<int>(storage->value_type())));
  }
}

OpKind map_op_kind(fbs::OpKind k) {
  switch (k) {
    case fbs::OpKind::CALL_FUNCTION:
      return OpKind::CallFunction;
    case fbs::OpKind::PLACEHOLDER:
      return OpKind::Placeholder;
    case fbs::OpKind::OUTPUT:
      return OpKind::Output;
    default:
      throw std::runtime_error(
          "build_graph: unsupported OpKind value " +
          std::to_string(static_cast<int>(k)));
  }
}

OutputValueKind map_output_value_kind(fbs::OutputValueKind k) {
  switch (k) {
    case fbs::OutputValueKind::TENSOR:
      return OutputValueKind::Tensor;
    case fbs::OutputValueKind::TENSOR_LIST:
      return OutputValueKind::TensorList;
    case fbs::OutputValueKind::INT:
      return OutputValueKind::Int;
    case fbs::OutputValueKind::BOOL:
      return OutputValueKind::Bool;
    case fbs::OutputValueKind::FLOAT:
      return OutputValueKind::Float;
    default:
      throw std::runtime_error(
          "build_graph: unsupported OutputValueKind value " +
          std::to_string(static_cast<int>(k)));
  }
}

ValueRole map_input_kind(fbs::InputKind k) {
  switch (k) {
    case fbs::InputKind::USER_INPUT:
      return ValueRole::UserInput;
    case fbs::InputKind::PARAMETER:
      return ValueRole::Parameter;
    case fbs::InputKind::BUFFER:
      return ValueRole::Buffer;
    case fbs::InputKind::CONSTANT_TENSOR:
      return ValueRole::ConstantTensor;
    default:
      throw std::runtime_error(
          "build_method: unsupported InputKind value " +
          std::to_string(static_cast<int>(k)));
  }
}

OutputKind map_output_kind(fbs::OutputKind k) {
  switch (k) {
    case fbs::OutputKind::USER_OUTPUT:
      return OutputKind::UserOutput;
    case fbs::OutputKind::BUFFER_MUTATION:
      return OutputKind::BufferMutation;
    case fbs::OutputKind::USER_INPUT_MUTATION:
      return OutputKind::UserInputMutation;
    default:
      throw std::runtime_error(
          "build_method: unsupported OutputKind value " +
          std::to_string(static_cast<int>(k)));
  }
}

// The wire describes a dim as a min..max range, while the IR holds a concrete
// extent. Collapsing a range to its upper bound would run the graph at that
// bound and compute over elements the caller never supplied, so a dim that is
// not a single non-negative extent is refused where it enters the IR.
int64_t static_extent(
    const fbs::Dim* d,
    const std::string& value_name,
    flatbuffers::uoffset_t i) {
  if (d->min() == d->max() && d->min() >= 0) {
    return d->min();
  }
  throw std::runtime_error(
      "build_tensor_meta: " + value_name + " dim " + std::to_string(i) +
      " is not a static extent (" + std::to_string(d->min()) + ".." +
      (d->max() < 0 ? std::string("inf") : std::to_string(d->max())) +
      "); this runtime requires static shapes");
}

bool range_fits_dense_dtype(int64_t lower, int64_t upper, ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Byte:
      return lower >= 0 && upper <= UINT8_MAX;
    case ScalarType::Char:
      return lower >= INT8_MIN && upper <= INT8_MAX;
    case ScalarType::Short:
      return lower >= INT16_MIN && upper <= INT16_MAX;
    case ScalarType::Int:
      return lower >= INT32_MIN && upper <= INT32_MAX;
    case ScalarType::Long:
      return true;
    case ScalarType::UInt16:
      return lower >= 0 && upper <= UINT16_MAX;
    case ScalarType::UInt32:
      return lower >= 0 && upper <= UINT32_MAX;
    case ScalarType::UInt64:
      return lower >= 0;
    case ScalarType::Half:
    case ScalarType::Float:
    case ScalarType::Double:
    case ScalarType::Bool:
    case ScalarType::BFloat16:
      return false;
  }
  return false;
}

// Largest finite value and smallest positive (subnormal) value of a float
// dtype.
std::pair<double, double> float_limits(ScalarType dtype) {
  if (dtype == ScalarType::Half) {
    return {65504.0, 0x1p-24};
  }
  if (dtype == ScalarType::BFloat16) {
    return {0x1.fep127, 0x1p-133};
  }
  if (dtype == ScalarType::Float) {
    return {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::denorm_min()};
  }
  return {
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::denorm_min()};
}

void validate_quant_param(const QuantParam& param, bool scale) {
  const ScalarType dtype =
      std::visit([](const auto& value) { return value.dtype; }, param);
  if (!is_float_type(dtype) && (scale || !is_integer_type(dtype))) {
    throw std::runtime_error(
        std::string("build_tensor_meta: invalid ") +
        (scale ? "scale" : "zero-point") + " dtype");
  }
  if (const auto* float_param = std::get_if<InlineFloatQuantParam>(&param)) {
    const auto [max_value, min_positive] = float_limits(float_param->dtype);
    if (std::abs(float_param->value) > max_value ||
        (scale && float_param->value < min_positive)) {
      throw std::runtime_error(
          std::string("build_tensor_meta: inline ") +
          (scale ? "scale is not finite and positive"
                 : "zero point is not finite") +
          " in its dtype");
    }
  } else if (const auto* int_param = std::get_if<InlineIntQuantParam>(&param);
             int_param != nullptr &&
             !range_fits_dense_dtype(
                 int_param->value, int_param->value, int_param->dtype)) {
    throw std::runtime_error(
        "build_tensor_meta: inline zero point does not fit its dtype");
  }
}

void validate_affine_quantization(
    const AffineQuantization& quant,
    const TensorMeta& meta) {
  if (!is_float_type(quant.expressed_dtype)) {
    throw std::runtime_error(
        "build_tensor_meta: affine expressed dtype must be floating point");
  }
  if (quant.quant_min >= quant.quant_max) {
    throw std::runtime_error(
        "build_tensor_meta: affine quant_min must be less than quant_max");
  }
  if (quant.block_shape.size() != meta.sizes.size()) {
    throw std::runtime_error(
        "build_tensor_meta: affine block_shape rank does not match tensor rank");
  }
  if (std::any_of(
          quant.block_shape.begin(),
          quant.block_shape.end(),
          [](const int64_t block) { return block < 0; })) {
    throw std::runtime_error(
        "build_tensor_meta: affine block_shape entries must be non-negative");
  }
  validate_quant_param(quant.scale, true);
  if (quant.zero_point.has_value()) {
    validate_quant_param(*quant.zero_point, false);
    if (const auto* value =
            std::get_if<InlineIntQuantParam>(&*quant.zero_point);
        value != nullptr &&
        (value->value < quant.quant_min || value->value > quant.quant_max)) {
      throw std::runtime_error(
          "build_tensor_meta: inline zero point is outside affine range");
    }
  }

  if (std::holds_alternative<DenseQuantizedStorage>(quant.storage)) {
    if (!range_fits_dense_dtype(quant.quant_min, quant.quant_max, meta.dtype)) {
      throw std::runtime_error(
          "build_tensor_meta: affine range does not fit dense storage dtype");
    }
    return;
  }

  const auto& storage = std::get<PackedBitsQuantizedStorage>(quant.storage);
  if (meta.dtype != ScalarType::Byte || storage.bit_width < 1 ||
      storage.bit_width >= 8) {
    throw std::runtime_error(
        "build_tensor_meta: invalid packed-bit affine storage");
  }
  const int64_t code_count = int64_t{1} << storage.bit_width;
  switch (storage.signed_encoding) {
    case QuantSignedEncoding::Unsigned:
      if (storage.storage_offset != 0 || quant.quant_min < 0 ||
          quant.quant_max >= code_count) {
        throw std::runtime_error(
            "build_tensor_meta: invalid unsigned packed affine range");
      }
      break;
    case QuantSignedEncoding::TwosComplement: {
      const int64_t lower = -(int64_t{1} << (storage.bit_width - 1));
      const int64_t upper = (int64_t{1} << (storage.bit_width - 1)) - 1;
      if (storage.storage_offset != 0 || quant.quant_min < lower ||
          quant.quant_max > upper) {
        throw std::runtime_error(
            "build_tensor_meta: invalid two's-complement packed affine range");
      }
      break;
    }
    case QuantSignedEncoding::Offset:
      if (quant.quant_min < storage.storage_offset ||
          static_cast<uint64_t>(quant.quant_max) -
                  static_cast<uint64_t>(storage.storage_offset) >=
              static_cast<uint64_t>(code_count)) {
        throw std::runtime_error(
            "build_tensor_meta: invalid offset packed affine range");
      }
      break;
  }
}

Quantization build_quantization(
    const fbs::QuantSpec* spec,
    const TensorMeta& meta) {
  if (spec == nullptr || spec->scheme() == nullptr) {
    throw std::runtime_error("build_tensor_meta: missing quantization spec");
  }
  switch (spec->scheme_type()) {
    case fbs::QuantScheme::AffineQuantization: {
      const auto* value = spec->scheme_as_AffineQuantization();
      AffineQuantization out;
      out.expressed_dtype = map_scalar_type(value->expressed_dtype());
      out.quant_min = value->quant_min();
      out.quant_max = value->quant_max();
      for (const int64_t block : *value->block_shape()) {
        out.block_shape.push_back(block);
      }
      out.scale = build_quant_param(value->scale());
      if (value->zero_point() != nullptr) {
        out.zero_point = build_quant_param(value->zero_point());
      }
      out.rounding = map_quant_rounding_mode(value->rounding());
      out.storage = build_quantized_storage(value->storage());
      validate_affine_quantization(out, meta);
      return out;
    }
    case fbs::QuantScheme::OpaqueQuantization: {
      OpaqueQuantization out;
      out.codec = str_of(spec->scheme_as_OpaqueQuantization()->codec());
      if (out.codec.empty()) {
        throw std::runtime_error(
            "build_tensor_meta: OpaqueQuantization codec must not be empty");
      }
      if (meta.dtype != ScalarType::Byte) {
        throw std::runtime_error(
            "build_tensor_meta: OpaqueQuantization requires BYTE dtype");
      }
      return out;
    }
    case fbs::QuantScheme::NONE:
    default:
      throw std::runtime_error(
          "build_tensor_meta: unsupported QuantScheme value " +
          std::to_string(static_cast<int>(spec->scheme_type())));
  }
}

TensorMeta build_tensor_meta(
    const fbs::TensorMeta* m,
    const std::string& name) {
  TensorMeta out;
  if (m == nullptr) {
    return out;
  }
  out.dtype = map_scalar_type(m->dtype());
  if (const auto* sizes = m->sizes()) {
    out.sizes.reserve(sizes->size());
    for (flatbuffers::uoffset_t i = 0; i < sizes->size(); ++i) {
      out.sizes.push_back(static_extent(sizes->Get(i), name, i));
    }
  }
  if (const auto* dord = m->dim_order()) {
    out.dim_order_hint.reserve(dord->size());
    for (flatbuffers::uoffset_t i = 0; i < dord->size(); ++i) {
      out.dim_order_hint.push_back(static_cast<int32_t>(dord->Get(i)));
    }
  }
  if (m->quant() != nullptr) {
    out.quantization = build_quantization(m->quant(), out);
  }
  return out;
}

// value name -> tensor metadata.
using MetaTable = std::unordered_map<std::string, const fbs::TensorMeta*>;

// One graph body plus the SSA-name -> value-id map used to build it.
//
// The wire format addresses values two ways: a graph body is positional, and
// the value list keeps that (a ValueId is an index), while the method-level
// side tables -- constants, mutable_buffers, output_specs -- name their targets
// by SSA string, since the serializer writes them independently of the body.
// Only the builder knows how one maps to the other, so it hands the map back
// for build_method to resolve those names against.
//
// Build-time scaffolding: nothing outside build_method sees it, and neither
// Method nor Graph stores it. Once the bindings are resolved to ids it is
// discarded, and the graph is index-addressed from then on.
struct BuiltGraph {
  Graph graph;
  std::unordered_map<std::string, ValueId> name_to_id;
};

// `extra_meta` supplies metadata for values the graph's own tensor_values side
// table omits: a constant placeholder's meta rides on the Method's
// NamedTensorRef binding instead, so build_method passes it in to type those
// values on creation. Subgraphs have no such bindings.
BuiltGraph build_graph(const fbs::Graph* g, const MetaTable& extra_meta = {});

// Builds one Graph body. The graph under construction, its SSA-name -> id map,
// and the name -> metadata table are shared by every step of the build, so they
// are members rather than threaded through each call. One builder per body: a
// subgraph gets a fresh one, which is what gives each body its own independent
// SSA namespace.
class GraphBuilder {
 public:
  BuiltGraph run(const fbs::Graph* g, const MetaTable& extra_meta);

 private:
  // Resolve a name to its ValueId, creating the Value on first mention.
  // A name in the meta side table becomes a Tensor value; otherwise a None
  // value (scalar / symbolic outputs, refined once the loader models them).
  ValueId id_of(const std::string& name);

  Argument convert_arg(const fbs::Argument* a);

  Graph graph_;
  std::unordered_map<std::string, ValueId> n2i_;
  MetaTable tm_;
};

ValueId GraphBuilder::id_of(const std::string& name) {
  if (name.empty()) {
    return kInvalid;
  }
  const auto it = n2i_.find(name);
  if (it != n2i_.end()) {
    return it->second;
  }
  const ValueId id = static_cast<ValueId>(graph_.values.size());
  const auto mit = tm_.find(name);
  if (mit != tm_.end() && mit->second != nullptr) {
    graph_.values.emplace_back(name, build_tensor_meta(mit->second, name));
  } else {
    graph_.values.emplace_back(name);
  }
  n2i_[name] = id;
  return id;
}

Argument GraphBuilder::convert_arg(const fbs::Argument* a) {
  using AV = fbs::ArgumentValue;
  switch (a->value_type()) {
    case AV::NONE:
    case AV::NoneArg:
      return NoneArg{};
    case AV::TensorArg: {
      TensorArg t;
      t.id = id_of(str_of(a->value_as_TensorArg()->name()));
      return t;
    }
    case AV::IntArg: {
      const auto* x = a->value_as_IntArg();
      IntArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::FloatArg: {
      const auto* x = a->value_as_FloatArg();
      FloatArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::BoolArg: {
      const auto* x = a->value_as_BoolArg();
      BoolArg r;
      r.value = x->value();
      r.id = nonempty(x->ref()) ? id_of(x->ref()->str()) : kInvalid;
      return r;
    }
    case AV::StringArg: {
      StringArg r;
      r.value = str_of(a->value_as_StringArg()->value());
      return r;
    }
    case AV::ScalarTypeArg: {
      ScalarTypeArg r;
      r.value = map_scalar_type(a->value_as_ScalarTypeArg()->value());
      return r;
    }
    case AV::IntListArg: {
      const auto* x = a->value_as_IntListArg();
      IntListArg r;
      if (const auto* vals = x->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      if (const auto* refs = x->refs()) {
        for (flatbuffers::uoffset_t i = 0; i < refs->size(); ++i) {
          r.ids.push_back(
              nonempty(refs->Get(i)) ? id_of(refs->Get(i)->str()) : kInvalid);
        }
      }
      return r;
    }
    case AV::FloatListArg: {
      FloatListArg r;
      if (const auto* vals = a->value_as_FloatListArg()->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      return r;
    }
    case AV::BoolListArg: {
      BoolListArg r;
      if (const auto* vals = a->value_as_BoolListArg()->values()) {
        for (flatbuffers::uoffset_t i = 0; i < vals->size(); ++i) {
          r.values.push_back(vals->Get(i));
        }
      }
      return r;
    }
    case AV::TensorListArg: {
      TensorListArg r;
      if (const auto* nm = a->value_as_TensorListArg()->names()) {
        for (flatbuffers::uoffset_t i = 0; i < nm->size(); ++i) {
          r.ids.push_back(id_of(nm->Get(i)->str()));
        }
      }
      return r;
    }
    case AV::OptionalTensorListArg: {
      const auto* oa = a->value_as_OptionalTensorListArg();
      const auto* nm = oa->names();
      const auto* hv = oa->has_value();
      OptionalTensorListArg r;
      if (nm != nullptr) {
        for (flatbuffers::uoffset_t i = 0; i < nm->size(); ++i) {
          const bool present = hv != nullptr && i < hv->size() && hv->Get(i);
          r.ids.push_back(present ? id_of(nm->Get(i)->str()) : kInvalid);
        }
      }
      return r;
    }
    case AV::GraphArg: {
      const fbs::GraphArg* ga = a->value_as_GraphArg();
      GraphArg r;
      r.name = str_of(ga->name());
      r.subgraph_id = static_cast<GraphId>(graph_.subgraphs.size());
      graph_.subgraphs.push_back(build_graph(ga->graph()).graph);
      return r;
    }
    default:
      throw std::runtime_error(
          "build_graph: unsupported ArgumentValue value " +
          std::to_string(static_cast<int>(a->value_type())));
  }
}

BuiltGraph GraphBuilder::run(const fbs::Graph* g, const MetaTable& extra_meta) {
  if (g == nullptr) {
    return {};
  }

  if (const auto* tvs = g->tensor_values()) {
    for (flatbuffers::uoffset_t i = 0; i < tvs->size(); ++i) {
      const fbs::TensorValue* tv = tvs->Get(i);
      tm_[str_of(tv->name())] = tv->meta();
    }
  }
  for (const auto& entry : extra_meta) {
    const fbs::TensorMeta*& slot = tm_[entry.first];
    if (slot == nullptr) {
      slot = entry.second;
    }
  }

  // Pre-create meta-carrying values in a deterministic order (nicer ids).
  if (const auto* tvs = g->tensor_values()) {
    for (flatbuffers::uoffset_t i = 0; i < tvs->size(); ++i) {
      id_of(str_of(tvs->Get(i)->name()));
    }
  }

  if (const auto* nodes = g->nodes()) {
    for (flatbuffers::uoffset_t i = 0; i < nodes->size(); ++i) {
      const fbs::Node* nd = nodes->Get(i);
      Node node;
      node.name = str_of(nd->name());
      node.op_kind = map_op_kind(nd->op_kind());
      node.target = str_of(nd->target());

      if (const auto* ins = nd->inputs()) {
        for (flatbuffers::uoffset_t j = 0; j < ins->size(); ++j) {
          const fbs::NamedArgument* na = ins->Get(j);
          NamedArgument narg;
          narg.name = str_of(na->name());
          narg.mutated = na->mutated();
          narg.arg = convert_arg(na->arg());
          node.inputs.push_back(std::move(narg));
        }
      }

      if (const auto* outs = nd->outputs()) {
        for (flatbuffers::uoffset_t j = 0; j < outs->size(); ++j) {
          const fbs::Output* o = outs->Get(j);
          Output out;
          out.kind = map_output_value_kind(o->kind());
          if (o->kind() == fbs::OutputValueKind::TENSOR_LIST) {
            if (const auto* nm = o->names()) {
              for (flatbuffers::uoffset_t k = 0; k < nm->size(); ++k) {
                out.elem_ids.push_back(id_of(nm->Get(k)->str()));
              }
            }
          } else {
            out.value_id = id_of(str_of(o->name()));
            if (nonempty(o->alias_of()) && valid(out.value_id)) {
              const ValueId alias_id = id_of(o->alias_of()->str());
              if (alias_id == out.value_id) {
                throw std::runtime_error(
                    "build_graph: output '" + str_of(o->name()) +
                    "' cannot alias itself");
              }
              graph_.values.at(static_cast<size_t>(out.value_id)).alias_id =
                  alias_id;
            }
          }
          node.outputs.push_back(std::move(out));
        }
      }

      // A placeholder with no explicit Output still produces its named value;
      // synthesize one so def-use wiring records the placeholder as producer.
      if (node.op_kind == OpKind::Placeholder && node.outputs.empty() &&
          !node.name.empty()) {
        Output out;
        out.value_id = id_of(node.name);
        node.outputs.push_back(out);
      }

      graph_.nodes.push_back(std::move(node));
    }
  }

  if (const auto* gi = g->inputs()) {
    for (flatbuffers::uoffset_t i = 0; i < gi->size(); ++i) {
      graph_.input_ids.push_back(id_of(gi->Get(i)->str()));
    }
  }
  if (const auto* go = g->outputs()) {
    for (flatbuffers::uoffset_t i = 0; i < go->size(); ++i) {
      graph_.output_ids.push_back(id_of(go->Get(i)->str()));
    }
  }

  graph_.initialize_schedule();
  graph_.rebuild_def_use();
  return BuiltGraph{std::move(graph_), std::move(n2i_)};
}

BuiltGraph build_graph(const fbs::Graph* g, const MetaTable& extra_meta) {
  return GraphBuilder().run(g, extra_meta);
}

// Resolve a method-level binding name (namespace 2) against the top-level
// graph's SSA names, or kInvalid if the graph holds no such value.
ValueId id_of_name(
    const std::unordered_map<std::string, ValueId>& n2i,
    const std::string& name) {
  const auto it = n2i.find(name);
  return it != n2i.end() ? it->second : kInvalid;
}

ValueId require_binding_value(
    const std::unordered_map<std::string, ValueId>& n2i,
    const std::string& name) {
  const ValueId id = id_of_name(n2i, name);
  if (!valid(id)) {
    throw std::runtime_error(
        "build_method: data binding '" + name +
        "' does not name a graph value");
  }
  return id;
}

void stamp_role(Graph& graph, ValueId id, ValueRole role) {
  if (in_bounds(id, graph.values.size())) {
    graph.values.at(static_cast<size_t>(id)).role = role;
  }
}

} // namespace

Method Program::build_method(size_t index) const {
  if (program_fb_ == nullptr) {
    throw std::runtime_error("build_method: program is not loaded");
  }
  const auto* methods = program_fb_->methods();
  if (methods == nullptr || index >= methods->size()) {
    throw std::runtime_error("build_method: method index out of range");
  }
  const fbs::Method* m =
      methods->Get(static_cast<flatbuffers::uoffset_t>(index));

  Method method;
  method.name = str_of(m->name());

  MetaTable constant_meta;
  if (const auto* cs = m->constants()) {
    for (flatbuffers::uoffset_t i = 0; i < cs->size(); ++i) {
      const fbs::NamedTensorRef* c = cs->Get(i);
      constant_meta[str_of(c->name())] = c->meta();
    }
  }

  BuiltGraph built = build_graph(m->graph(), constant_meta);
  const std::unordered_map<std::string, ValueId>& n2i = built.name_to_id;
  method.graph = std::move(built.graph);
  Graph& graph = method.graph;

  // external-constant / buffer identity (key) -> value, for BufferMutation
  // output targets (a namespace-3 fqn, not an SSA name).
  std::unordered_map<std::string, ValueId> key_to_id;
  std::unordered_set<ValueId> bound_ids;

  if (const auto* cs = m->constants()) {
    for (flatbuffers::uoffset_t i = 0; i < cs->size(); ++i) {
      const fbs::NamedTensorRef* c = cs->Get(i);
      DataBinding b;
      const std::string name = str_of(c->name());
      b.value_id = require_binding_value(n2i, name);
      if (!bound_ids.insert(b.value_id).second) {
        throw std::runtime_error(
            "build_method: graph value '" + name +
            "' has multiple data bindings");
      }
      b.role = map_input_kind(c->kind());
      b.key = str_of(c->data_key());
      b.has_data = true;
      b.mutated = c->mutated();
      stamp_role(graph, b.value_id, b.role);
      if (!b.key.empty()) {
        key_to_id[b.key] = b.value_id;
      }
      method.data_bindings.push_back(std::move(b));
    }
  }

  if (const auto* mbs = m->mutable_buffers()) {
    for (flatbuffers::uoffset_t i = 0; i < mbs->size(); ++i) {
      const fbs::MutableBufferSpec* mb = mbs->Get(i);
      DataBinding b;
      const std::string name = str_of(mb->name());
      b.value_id = require_binding_value(n2i, name);
      if (!bound_ids.insert(b.value_id).second) {
        throw std::runtime_error(
            "build_method: graph value '" + name +
            "' has multiple data bindings");
      }
      b.role = ValueRole::Buffer;
      b.key = str_of(mb->fqn());
      b.has_data = false;
      b.mutated = true;
      stamp_role(graph, b.value_id, ValueRole::Buffer);
      if (!b.key.empty()) {
        key_to_id[b.key] = b.value_id;
      }
      method.data_bindings.push_back(std::move(b));
    }
  }

  // Top-level graph inputs not otherwise bound are user inputs.
  for (const ValueId id : graph.input_ids) {
    if (in_bounds(id, graph.values.size()) &&
        graph.values[id].role == ValueRole::Intermediate) {
      graph.values[id].role = ValueRole::UserInput;
    }
  }

  // output_specs are parallel to graph.outputs (same order); each classifies
  // graph.output_ids[i]. The mutation target resolves to a placeholder value:
  // an fqn (BufferMutation) via key_to_id, else an SSA name
  // (UserInputMutation).
  if (const auto* os = m->output_specs()) {
    if (os->size() != graph.output_ids.size()) {
      throw std::runtime_error(
          "build_method: output_specs count does not match graph outputs");
    }
    for (flatbuffers::uoffset_t i = 0; i < os->size(); ++i) {
      const fbs::OutputSpec* o = os->Get(i);
      OutputSpec spec;
      spec.kind = map_output_kind(o->kind());
      const std::string target = str_of(o->target());
      if (spec.kind != OutputKind::UserOutput) {
        if (spec.kind == OutputKind::BufferMutation) {
          const auto it = key_to_id.find(target);
          spec.target_id = it != key_to_id.end() ? it->second : kInvalid;
        } else {
          spec.target_id = id_of_name(n2i, target);
        }
        if (!valid(spec.target_id)) {
          throw std::runtime_error(
              "build_method: mutation target '" + target +
              "' does not name a bound value");
        }
      }
      method.output_specs.push_back(spec);
    }
  }

  return method;
}

// Defined here (rather than in Program.cpp) so it sits next to build_method and
// the deserializer helpers it drives: get_method is the public lazy entry
// point, build_method the private materializer it calls on a cache miss.
const Method& Program::get_method(const std::string& name) const {
  const auto it = method_cache_.find(name);
  if (it != method_cache_.end()) {
    return it->second;
  }
  if (program_fb_ != nullptr) {
    if (const auto* methods = program_fb_->methods()) {
      for (flatbuffers::uoffset_t i = 0; i < methods->size(); ++i) {
        const flatbuffers::String* method_name = methods->Get(i)->name();
        if (method_name != nullptr &&
            std::string_view(method_name->c_str(), method_name->size()) ==
                name) {
          auto res = method_cache_.emplace(name, build_method(i));
          return res.first->second;
        }
      }
    }
  }
  throw std::runtime_error(
      "Program::get_method: no method named '" + name + "'");
}

} // namespace ptn
