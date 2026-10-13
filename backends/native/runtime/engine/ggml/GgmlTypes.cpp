// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace ptn::ggml_detail {
namespace {

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

} // namespace

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

static ggml_type to_ggml_type(const TensorMeta& meta) {
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

} // namespace ptn::ggml_detail
