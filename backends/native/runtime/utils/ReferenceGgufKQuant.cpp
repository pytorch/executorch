// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/utils/ReferenceGgufKQuant.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ptn::reference {
namespace {

constexpr std::string_view kQ4K = "gguf:q4_k";
constexpr std::string_view kQ6K = "gguf:q6_k";
constexpr size_t kQ4KBlockBytes = 144;
constexpr size_t kQ6KBlockBytes = 210;
// Below this many weights per thread, starting threads costs more than it
// saves.
constexpr int64_t kMinWeightsPerThread = int64_t{1} << 20;

using BlockDecoder = void (*)(const uint8_t*, float*);

float half_to_float(const uint8_t* bytes) {
  const uint32_t h =
      static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8);
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1Fu;
  const uint32_t mantissa = h & 0x3FFu;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return sign != 0 ? -magnitude : magnitude;
  }
  if (exponent == 0x1Fu) {
    return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
  }
  return std::bit_cast<float>(
      sign | ((exponent + 112) << 23) | (mantissa << 13));
}

// The 12 scale bytes hold eight 6-bit (scale, min) pairs.
void q4_k_scale_min(const uint8_t* scales, int pair, float* scale, float* min) {
  if (pair < 4) {
    *scale = scales[pair] & 63;
    *min = scales[pair + 4] & 63;
  } else {
    *scale = (scales[pair + 4] & 0xF) | ((scales[pair - 4] >> 6) << 4);
    *min = (scales[pair + 4] >> 4) | ((scales[pair] >> 6) << 4);
  }
}

// block_q4_K: fp16 d, fp16 dmin, 12 scale bytes, 128 nibble bytes. Each
// 32-byte group yields 32 weights from its low nibbles, then 32 from its high.
void decode_q4_k_block(const uint8_t* block, float* y) {
  const float d = half_to_float(block);
  const float dmin = half_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* q = block + 16;
  for (int group = 0; group < 4; ++group) {
    float scale = 0;
    float min = 0;
    q4_k_scale_min(scales, 2 * group, &scale, &min);
    const float d1 = d * scale;
    const float m1 = dmin * min;
    q4_k_scale_min(scales, 2 * group + 1, &scale, &min);
    const float d2 = d * scale;
    const float m2 = dmin * min;
    const uint8_t* qg = q + 32 * group;
    float* yg = y + 64 * group;
    for (int l = 0; l < 32; ++l) {
      yg[l] = d1 * static_cast<float>(qg[l] & 0xF) - m1;
      yg[l + 32] = d2 * static_cast<float>(qg[l] >> 4) - m2;
    }
  }
}

// block_q6_K: 128 low-nibble bytes, 64 high-bit bytes, 16 int8 scales, fp16 d.
// Each 128-weight half reads 64 ql and 32 qh bytes, four weights 32 apart per
// position, with one scale per 16 weights.
void decode_q6_k_block(const uint8_t* block, float* y) {
  const auto* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = half_to_float(block + 208);
  for (int half = 0; half < 2; ++half) {
    const uint8_t* ql = block + 64 * half;
    const uint8_t* qh = block + 128 + 32 * half;
    const int8_t* sc = scales + 8 * half;
    float* yh = y + 128 * half;
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
      const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      yh[l] = d * sc[is] * q1;
      yh[l + 32] = d * sc[is + 2] * q2;
      yh[l + 64] = d * sc[is + 4] * q3;
      yh[l + 96] = d * sc[is + 6] * q4;
    }
  }
}

struct Codec {
  size_t block_bytes;
  BlockDecoder decode_block;
};

Codec codec_of(std::string_view codec) {
  if (codec == kQ4K) {
    return {kQ4KBlockBytes, decode_q4_k_block};
  }
  if (codec == kQ6K) {
    return {kQ6KBlockBytes, decode_q6_k_block};
  }
  throw std::invalid_argument(
      "unsupported GGUF K-quant codec '" + std::string(codec) + "'");
}

int64_t blocks_per_row(int64_t cols) {
  if (cols <= 0 || cols % kGgufKQuantBlockElements != 0) {
    throw std::invalid_argument(
        "GGUF K-quant rows must hold a positive multiple of 256 weights");
  }
  return cols / kGgufKQuantBlockElements;
}

void decode_rows(
    const Codec& codec,
    const uint8_t* data,
    int64_t first,
    int64_t last,
    int64_t blocks,
    float* out) {
  for (int64_t r = first; r < last; ++r) {
    const uint8_t* row = data + r * blocks * codec.block_bytes;
    float* y = out + r * blocks * kGgufKQuantBlockElements;
    for (int64_t b = 0; b < blocks; ++b) {
      codec.decode_block(
          row + b * codec.block_bytes, y + b * kGgufKQuantBlockElements);
    }
  }
}

} // namespace

// cppcheck-suppress unusedFunction
bool is_gguf_kquant(std::string_view codec) {
  return codec == kQ4K || codec == kQ6K;
}

// cppcheck-suppress unusedFunction
size_t gguf_kquant_row_nbytes(std::string_view codec, int64_t cols) {
  return static_cast<size_t>(blocks_per_row(cols)) *
      codec_of(codec).block_bytes;
}

// cppcheck-suppress unusedFunction
void decode_gguf_kquant(
    std::string_view codec,
    const uint8_t* data,
    size_t nbytes,
    int64_t rows,
    int64_t cols,
    float* out,
    unsigned max_threads) {
  const Codec c = codec_of(codec);
  const int64_t blocks = blocks_per_row(cols);
  size_t expected = 0;
  if (rows < 0 ||
      __builtin_mul_overflow(
          static_cast<size_t>(rows),
          static_cast<size_t>(blocks) * c.block_bytes,
          &expected) ||
      nbytes != expected) {
    throw std::invalid_argument(
        "GGUF K-quant data does not hold " + std::to_string(rows) +
        " rows of " + std::to_string(cols) + " weights");
  }
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  const int64_t by_work =
      std::max<int64_t>(1, rows * cols / kMinWeightsPerThread);
  const int64_t threads = std::min<int64_t>(
      {max_threads == 0 ? hardware : max_threads,
       by_work,
       std::max<int64_t>(rows, 1)});
  if (threads <= 1) {
    decode_rows(c, data, 0, rows, blocks, out);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(threads);
  try {
    for (int64_t t = 0; t < threads; ++t) {
      workers.emplace_back(
          decode_rows,
          c,
          data,
          rows * t / threads,
          rows * (t + 1) / threads,
          blocks,
          out);
    }
  } catch (...) {
    for (std::thread& worker : workers) {
      worker.join();
    }
    throw;
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
}

} // namespace ptn::reference
