// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include <executorch/backends/native/runtime/deserialize/JsonParser.h>

namespace ptn {
namespace {

// Reserved header member holding free-form string metadata, not a tensor.
constexpr std::string_view kMetadataKey = "__metadata__";
constexpr size_t kHeaderLenSize = 8;

struct DtypeCode {
  std::string_view code;
  ScalarType dtype;
};

// safetensors dtype codes, as written by safetensors.torch. Codes with no
// ScalarType counterpart (complex, 4-bit and 8-bit float variants) are absent
// and rejected by name, so an unsupported constant fails at load rather than
// being misread as another width.
constexpr std::array<DtypeCode, 13> kDtypeCodes{{
    {"F64", kDouble},
    {"F32", kFloat},
    {"F16", kHalf},
    {"BF16", kBFloat16},
    {"I64", kLong},
    {"I32", kInt},
    {"I16", kShort},
    {"I8", kChar},
    {"U8", kByte},
    {"BOOL", kBool},
    {"U16", kUInt16},
    {"U32", kUInt32},
    {"U64", kUInt64},
}};

ScalarType scalar_type_of(std::string_view code) {
  const auto it = std::ranges::find(kDtypeCodes, code, &DtypeCode::code);
  if (it == kDtypeCodes.end()) {
    throw std::runtime_error(
        "safetensors: unsupported dtype code: " + std::string(code));
  }
  return it->dtype;
}

uint64_t read_header_len(ByteSpan blob) {
  static_assert(
      std::endian::native == std::endian::little,
      "the length prefix is little-endian; a big-endian host needs a swap");
  if (blob.size() < kHeaderLenSize) {
    throw std::runtime_error(
        "safetensors: blob is shorter than its length prefix");
  }
  uint64_t len = 0;
  std::memcpy(&len, blob.data(), kHeaderLenSize);
  return len;
}

const JsonValue& required_member(
    const JsonValue& entry,
    std::string_view key,
    const std::string& name) {
  const JsonValue* value = entry.find(key);
  if (value == nullptr) {
    std::string message = "safetensors: entry '";
    message += name;
    message += "' has no '";
    message += key;
    message += "'";
    throw std::runtime_error(message);
  }
  return *value;
}

std::vector<int64_t> read_sizes(
    const JsonValue& shape,
    const std::string& name) {
  std::vector<int64_t> sizes;
  for (const JsonValue& dim : shape.as_array()) {
    const uint64_t value = dim.as_number();
    if (value > static_cast<uint64_t>(INT64_MAX)) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' has an out-of-range dimension");
    }
    sizes.push_back(static_cast<int64_t>(value));
  }
  return sizes;
}

// Element count of `sizes`, rejecting an overflowing product. A rank-0 shape is
// a scalar, whose element count is 1.
size_t numel_of(const std::vector<int64_t>& sizes, const std::string& name) {
  size_t numel = 1;
  for (const int64_t dim : sizes) {
    if (dim < 0) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' has a negative dimension");
    }
    const size_t d = static_cast<size_t>(dim);
    if (d != 0 && numel > SIZE_MAX / d) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' element count overflows");
    }
    numel *= d;
  }
  return numel;
}

} // namespace

SafeTensorsReader SafeTensorsReader::open(ByteSpan blob) {
  const uint64_t header_len = read_header_len(blob);
  if (header_len > blob.size() - kHeaderLenSize) {
    throw std::runtime_error("safetensors: header length exceeds the blob");
  }

  const std::string_view header_text(
      reinterpret_cast<const char*>(blob.data() + kHeaderLenSize),
      static_cast<size_t>(header_len));
  const JsonValue header = json_parse(header_text);
  if (!header.is_object()) {
    throw std::runtime_error("safetensors: header is not a JSON object");
  }

  SafeTensorsReader out;
  out.data_ = blob.subspan(kHeaderLenSize + static_cast<size_t>(header_len));

  for (const JsonMember& member : header.as_object()) {
    const std::string& name = member.key;
    if (name == kMetadataKey) {
      continue;
    }
    const JsonValue& entry = member.value;
    if (!entry.is_object()) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' is not an object");
    }

    TensorEntry parsed;
    parsed.dtype =
        scalar_type_of(required_member(entry, "dtype", name).as_string());
    parsed.sizes = read_sizes(required_member(entry, "shape", name), name);

    const JsonValue::Array& range =
        required_member(entry, "data_offsets", name).as_array();
    if (range.size() != 2) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' data_offsets is not a pair");
    }
    const uint64_t begin = range[0].as_number();
    const uint64_t end = range[1].as_number();
    if (begin > end || end > out.data_.size()) {
      throw std::runtime_error(
          "safetensors: entry '" + name +
          "' byte range is outside the data section");
    }
    parsed.offset = static_cast<size_t>(begin);
    parsed.nbytes = static_cast<size_t>(end - begin);

    // The payload must be exactly as large as its dtype and shape imply.
    // Without this, a short entry becomes an out-of-bounds read in whatever
    // consumes it, sized from the metadata rather than the bytes.
    const size_t expected =
        numel_of(parsed.sizes, name) * element_size(parsed.dtype);
    if (parsed.nbytes != expected) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' holds " +
          std::to_string(parsed.nbytes) +
          " bytes but its dtype and shape need " + std::to_string(expected));
    }

    if (!out.entries_.emplace(name, std::move(parsed)).second) {
      throw std::runtime_error("safetensors: duplicate entry: " + name);
    }
    out.names_.push_back(name);
  }

  return out;
}

const TensorEntry* SafeTensorsReader::find(const std::string& name) const {
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

ByteSpan SafeTensorsReader::bytes(const TensorEntry& entry) const {
  return data_.subspan(entry.offset, entry.nbytes);
}

size_t SafeTensorsReader::total_bytes() const {
  return std::accumulate(
      entries_.begin(),
      entries_.end(),
      size_t{0},
      [](size_t total, const auto& entry) {
        return total + entry.second.nbytes;
      });
}

} // namespace ptn
