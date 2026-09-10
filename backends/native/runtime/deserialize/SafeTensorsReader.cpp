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

#include <executorch/backends/native/runtime/deserialize/Json.h>

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

const Json& required_member(
    const Json& entry,
    std::string_view key,
    const std::string& name) {
  const auto value = entry.find(key);
  if (value == entry.end()) {
    std::string message = "safetensors: entry '";
    message += name;
    message += "' has no '";
    message += key;
    message += "'";
    throw std::runtime_error(message);
  }
  return value.value();
}

std::vector<int64_t> read_sizes(const Json& shape, const std::string& name) {
  if (!shape.is_array()) {
    throw std::runtime_error(
        "safetensors: entry '" + name + "' shape is not an array");
  }
  std::vector<int64_t> sizes;
  for (const Json& dim : shape) {
    if (!dim.is_number_unsigned()) {
      throw std::runtime_error(
          "safetensors: entry '" + name +
          "' has a non-negative integer dimension");
    }
    const uint64_t value = dim.get<uint64_t>();
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
  Json header;
  try {
    header = Json::parse(header_text);
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        "safetensors: invalid JSON header: " + std::string(error.what()));
  }
  if (!header.is_object()) {
    throw std::runtime_error("safetensors: header is not a JSON object");
  }

  SafeTensorsReader out;
  out.data_ = blob.subspan(kHeaderLenSize + static_cast<size_t>(header_len));

  for (auto member = header.begin(); member != header.end(); ++member) {
    const std::string& name = member.key();
    if (name == kMetadataKey) {
      if (!member.value().is_object() || !member.value().empty()) {
        throw std::runtime_error(
            "safetensors: __metadata__ must be an empty object");
      }
      continue;
    }
    const Json& entry = member.value();
    if (!entry.is_object()) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' is not an object");
    }

    TensorEntry parsed;
    const Json& dtype = required_member(entry, "dtype", name);
    if (!dtype.is_string()) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' dtype is not a string");
    }
    parsed.dtype = scalar_type_of(dtype.get_ref<const std::string&>());
    parsed.sizes = read_sizes(required_member(entry, "shape", name), name);

    const Json& range = required_member(entry, "data_offsets", name);
    if (!range.is_array() || range.size() != 2) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' data_offsets is not a pair");
    }
    if (!range[0].is_number_unsigned() || !range[1].is_number_unsigned()) {
      throw std::runtime_error(
          "safetensors: entry '" + name +
          "' data_offsets contains a non-negative integer");
    }
    const uint64_t begin = range[0].get<uint64_t>();
    const uint64_t end = range[1].get<uint64_t>();
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
    const size_t numel = numel_of(parsed.sizes, name);
    const size_t element_bytes = element_size(parsed.dtype);
    if (element_bytes != 0 && numel > SIZE_MAX / element_bytes) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' byte size overflows");
    }
    const size_t expected = numel * element_bytes;
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
