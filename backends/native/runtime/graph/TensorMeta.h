// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

enum class QuantRoundingMode : int8_t {
  ToNearestEven = 0,
  AwayFromZero = 1,
  TowardZero = 2,
  Floor = 3,
  Ceil = 4,
};

enum class QuantBitOrder : int8_t {
  LsbFirst = 0,
  MsbFirst = 1,
};

enum class QuantSignedEncoding : int8_t {
  Unsigned = 0,
  TwosComplement = 1,
  Offset = 2,
};

struct InlineFloatQuantParam {
  double value = 0;
  ScalarType dtype = ScalarType::Float;
  bool operator==(const InlineFloatQuantParam&) const = default;
};

struct InlineIntQuantParam {
  int64_t value = 0;
  ScalarType dtype = ScalarType::Long;
  bool operator==(const InlineIntQuantParam&) const = default;
};

struct ExternalQuantParam {
  std::string data_key;
  ScalarType dtype = ScalarType::Float;
  bool operator==(const ExternalQuantParam&) const = default;
};

using QuantParam = std::
    variant<InlineFloatQuantParam, InlineIntQuantParam, ExternalQuantParam>;

struct DenseQuantizedStorage {
  bool operator==(const DenseQuantizedStorage&) const = default;
};

struct PackedBitsQuantizedStorage {
  uint8_t bit_width = 0;
  QuantBitOrder bit_order = QuantBitOrder::LsbFirst;
  QuantSignedEncoding signed_encoding = QuantSignedEncoding::Unsigned;
  int64_t storage_offset = 0;
  bool operator==(const PackedBitsQuantizedStorage&) const = default;
};

using QuantizedStorage =
    std::variant<DenseQuantizedStorage, PackedBitsQuantizedStorage>;

struct AffineQuantization {
  ScalarType expressed_dtype = ScalarType::Float;
  int64_t quant_min = 0;
  int64_t quant_max = 0;
  std::vector<int64_t> block_shape;
  QuantParam scale;
  std::optional<QuantParam> zero_point;
  QuantRoundingMode rounding = QuantRoundingMode::ToNearestEven;
  QuantizedStorage storage;
  bool operator==(const AffineQuantization&) const = default;
};

struct OpaqueQuantization {
  std::string codec;
  bool operator==(const OpaqueQuantization&) const = default;
};

using Quantization = std::variant<AffineQuantization, OpaqueQuantization>;

// Logical tensor metadata: storage element type, shape, and optional
// quantization.
//
// sizes holds concrete extents. The wire format carries a per-dim range
// instead, but a runtime that plans and executes at fixed shapes cannot honor a
// dynamic dim, so deserialization rejects one rather than silently collapsing
// it to its upper bound.
//
// dim_order_hint is a permutation of dim indices, outermost first; empty means
// contiguous ([0, 1, ..., n-1]). It is a hint only for a tensor with no stored
// content — an activation — where an engine is free to pick its own physical
// layout. For a tensor whose bytes are serialized it instead describes the
// layout those bytes are actually in, and an engine that ignores it reads the
// weight wrong.
struct TensorMeta {
  ScalarType dtype = ScalarType::Float;
  std::vector<int64_t> sizes;
  std::vector<int32_t> dim_order_hint;
  std::optional<Quantization> quantization;

  size_t ndim() const {
    return sizes.size();
  }

  // True if dim_order_hint is empty or the identity permutation.
  bool is_contiguous() const;

  // Throws std::runtime_error on a negative extent, or on a count that
  // overflows int64_t.
  int64_t numel() const;

  // Exact on dim_order_hint: an empty hint and a spelled-out identity
  // permutation compare unequal though they mean the same layout.
  bool operator==(const TensorMeta&) const = default;
};

} // namespace ptn
