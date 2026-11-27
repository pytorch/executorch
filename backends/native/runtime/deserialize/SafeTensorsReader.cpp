// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <executorch/backends/native/runtime/deserialize/CheckedMath.h>
#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Json.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>

namespace ptn {
namespace {

// Reserved header member holding free-form string metadata, not a tensor.
constexpr std::string_view kMetadataKey = "__metadata__";
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
  if (shape.size() > detail::kMaxTensorRank) {
    throw ResourceLimitError(
        "safetensors: entry '" + name + "' exceeds tensor rank limit");
  }
  std::vector<int64_t> sizes;
  sizes.reserve(shape.size());
  for (const Json& dim : shape) {
    if (!dim.is_number_unsigned()) {
      throw std::runtime_error(
          "safetensors: entry '" + name +
          "' has a dimension that is not a non-negative integer");
    }
    const uint64_t value = dim.get<uint64_t>();
    if (value > detail::kMaxTensorDimension) {
      throw ResourceLimitError(
          "safetensors: entry '" + name + "' exceeds dimension limit");
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
    if (!detail::checked_mul(numel, static_cast<size_t>(dim), numel)) {
      throw ResourceLimitError(
          "safetensors: entry '" + name + "' element count overflows");
    }
  }
  return numel;
}

} // namespace

size_t SafeTensorsReader::header_size(ByteSpan prefix) {
  if (prefix.size() < kLengthPrefixSize) {
    throw std::runtime_error(
        "safetensors: blob is shorter than its length prefix");
  }
  uint64_t size = 0;
  for (size_t i = 0; i < kLengthPrefixSize; ++i) {
    size |= static_cast<uint64_t>(prefix[i]) << (8 * i);
  }
  if (size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("safetensors: header is too large");
  }
  return static_cast<size_t>(size);
}

SafeTensorsReader SafeTensorsReader::open(ByteSpan blob) {
  const size_t header_len = header_size(blob);
  if (header_len > detail::kMaxJsonBytes) {
    throw ResourceLimitError("safetensors: header exceeds size limit");
  }
  if (header_len > blob.size() - kLengthPrefixSize) {
    throw std::runtime_error("safetensors: header length exceeds the blob");
  }
  const ByteSpan header =
      blob.subspan(kLengthPrefixSize, static_cast<size_t>(header_len));
  return open_header(
      header,
      blob.size() - kLengthPrefixSize - static_cast<size_t>(header_len));
}

SafeTensorsReader SafeTensorsReader::open_header(
    ByteSpan header_bytes,
    size_t data_size) {
  if (header_bytes.size() > detail::kMaxJsonBytes) {
    throw ResourceLimitError("safetensors: header exceeds size limit");
  }
  const std::string_view header_text(
      reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
  Json header;
  try {
    header = parse_json(header_text);
  } catch (const Json::exception& error) {
    throw std::runtime_error(
        "safetensors: invalid JSON header: " + std::string(error.what()));
  }
  if (!header.is_object()) {
    throw std::runtime_error("safetensors: header is not a JSON object");
  }
  if (header.size() > detail::kMaxTensorCount) {
    throw ResourceLimitError("safetensors: tensor count exceeds limit");
  }

  SafeTensorsReader out;
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(header.size());

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
          "' data_offsets contains a value that is not a non-negative integer");
    }
    const uint64_t begin = range[0].get<uint64_t>();
    const uint64_t end = range[1].get<uint64_t>();
    if (begin > end || end > data_size ||
        end > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error(
          "safetensors: entry '" + name +
          "' byte range is outside the data section");
    }
    parsed.offset = static_cast<size_t>(begin);
    parsed.nbytes = static_cast<size_t>(end - begin);

    // The payload must be exactly as large as its dtype and shape imply.
    // Without this, a short entry becomes an out-of-bounds read in whatever
    // consumes it, sized from the metadata rather than the bytes.
    size_t expected = 0;
    if (!detail::checked_mul(
            numel_of(parsed.sizes, name),
            element_size(parsed.dtype),
            expected)) {
      throw ResourceLimitError(
          "safetensors: entry '" + name + "' byte size overflows");
    }
    if (parsed.nbytes != expected) {
      throw std::runtime_error(
          "safetensors: entry '" + name + "' holds " +
          std::to_string(parsed.nbytes) +
          " bytes but its dtype and shape need " + std::to_string(expected));
    }

    const auto [it, inserted] = out.entries_.emplace(name, std::move(parsed));
    if (!inserted) {
      throw std::runtime_error("safetensors: duplicate entry: " + name);
    }
    const TensorEntry& entry_info = it->second;
    ranges.emplace_back(entry_info.offset, static_cast<size_t>(end));
    if (!detail::checked_add(
            out.total_bytes_, entry_info.nbytes, out.total_bytes_)) {
      throw ResourceLimitError("safetensors: total payload size overflows");
    }
    if (out.total_bytes_ > detail::kMaxConstantBytes) {
      throw ResourceLimitError("safetensors: total payload exceeds size limit");
    }
    out.names_.push_back(name);
  }

  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i - 1].second > ranges[i].first) {
      throw std::runtime_error("safetensors: tensor byte ranges overlap");
    }
  }

  return out;
}

const TensorEntry* SafeTensorsReader::find(const std::string& name) const {
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

size_t SafeTensorsReader::total_bytes() const {
  return total_bytes_;
}

} // namespace ptn
