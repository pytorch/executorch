// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/utils/ReferenceGgufKQuant.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <ggml.h>
#include <gtest/gtest.h>

namespace ptn::reference {
namespace {

std::vector<float> decode_one_row(
    std::string_view codec,
    const std::vector<uint8_t>& block) {
  std::vector<float> out(kGgufKQuantBlockElements);
  decode_gguf_kquant(
      codec,
      block.data(),
      block.size(),
      1,
      kGgufKQuantBlockElements,
      out.data());
  return out;
}

// Packs eight 6-bit (scale, min) pairs the way GGUF's block_q4_K stores them:
// pairs 0-3 in the low six bits of bytes 0-3 and 4-7; pairs 4-7 as low
// nibbles in bytes 8-11 plus their top two bits in the top bits of 0-7.
std::array<uint8_t, 12> pack_q4_k_scales(
    const std::array<uint8_t, 8>& scales,
    const std::array<uint8_t, 8>& mins) {
  std::array<uint8_t, 12> packed{};
  for (int j = 0; j < 4; ++j) {
    packed[j] = scales[j] | ((scales[j + 4] >> 4) << 6);
    packed[j + 4] = mins[j] | ((mins[j + 4] >> 4) << 6);
    packed[j + 8] = (scales[j + 4] & 0xF) | ((mins[j + 4] & 0xF) << 4);
  }
  return packed;
}

TEST(ReferenceGgufKQuantTest, DecodesQ4KBlock) {
  const std::array<uint8_t, 8> scales = {1, 2, 3, 4, 17, 5, 33, 63};
  const std::array<uint8_t, 8> mins = {0, 1, 2, 3, 4, 18, 5, 40};
  std::vector<uint8_t> block = {0x00, 0x3C, 0x00, 0x38}; // d = 1, dmin = 0.5
  const std::array<uint8_t, 12> packed = pack_q4_k_scales(scales, mins);
  block.insert(block.end(), packed.begin(), packed.end());
  // Every group gets its own nibbles, so reading the wrong group fails.
  const auto low = [](int group, int l) { return (l + 5 * group) % 16; };
  const auto high = [](int group, int l) {
    return (15 - l % 16 + 3 * group) % 16;
  };
  for (int group = 0; group < 4; ++group) {
    for (int l = 0; l < 32; ++l) {
      block.push_back(
          static_cast<uint8_t>(low(group, l) | (high(group, l) << 4)));
    }
  }
  ASSERT_EQ(block.size(), 144);

  std::vector<float> expected;
  for (int group = 0; group < 4; ++group) {
    for (int l = 0; l < 32; ++l) {
      expected.push_back(
          scales[2 * group] * low(group, l) - 0.5f * mins[2 * group]);
    }
    for (int l = 0; l < 32; ++l) {
      expected.push_back(
          scales[2 * group + 1] * high(group, l) - 0.5f * mins[2 * group + 1]);
    }
  }
  EXPECT_EQ(decode_one_row("gguf:q4_k", block), expected);
}

TEST(ReferenceGgufKQuantTest, DecodesQ6KBlock) {
  std::array<int8_t, 16> scales{};
  for (int i = 0; i < 16; ++i) {
    scales[i] = static_cast<int8_t>(i - 8);
  }
  // Weight e stores the unsigned 6-bit value u and decodes to u - 32. Each
  // 128-weight half splits its low nibbles over 64 ql bytes and its top bits
  // over 32 qh bytes, four weights 32 apart per ql/qh position. The two halves
  // hold different values, so reading the wrong half fails.
  std::vector<uint8_t> ql(128);
  std::vector<uint8_t> qh(64);
  std::vector<float> expected(256);
  const float d = -0.25f;
  for (int e = 0; e < 256; ++e) {
    const int u = (e * 7 + (e / 128) * 13) % 64;
    const int half = e / 128;
    const int k = (e % 128) / 32;
    const int l = e % 32;
    uint8_t& low = ql[half * 64 + l + (k % 2) * 32];
    low |= static_cast<uint8_t>((u & 0xF) << (k / 2 * 4));
    qh[half * 32 + l] |= static_cast<uint8_t>((u >> 4) << (2 * k));
    expected[e] = d * scales[e / 16] * (u - 32);
  }
  std::vector<uint8_t> block = ql;
  block.insert(block.end(), qh.begin(), qh.end());
  for (const int8_t scale : scales) {
    block.push_back(static_cast<uint8_t>(scale));
  }
  block.push_back(0x00); // d = -0.25 (fp16 0xB400)
  block.push_back(0xB4);
  ASSERT_EQ(block.size(), 210);

  EXPECT_EQ(decode_one_row("gguf:q6_k", block), expected);
}

TEST(ReferenceGgufKQuantTest, SizesWholeSuperBlockRows) {
  EXPECT_TRUE(is_gguf_kquant("gguf:q4_k"));
  EXPECT_TRUE(is_gguf_kquant("gguf:q6_k"));
  EXPECT_FALSE(is_gguf_kquant("gguf:q4_0"));
  EXPECT_EQ(gguf_kquant_row_nbytes("gguf:q4_k", 512), 288);
  EXPECT_EQ(gguf_kquant_row_nbytes("gguf:q6_k", 256), 210);
  EXPECT_THROW(gguf_kquant_row_nbytes("gguf:q4_k", 255), std::invalid_argument);
  EXPECT_THROW(gguf_kquant_row_nbytes("gguf:q4_k", 0), std::invalid_argument);
  EXPECT_THROW(
      gguf_kquant_row_nbytes("gguf:q4_k", -256), std::invalid_argument);
  EXPECT_THROW(gguf_kquant_row_nbytes("gguf:q8_0", 256), std::invalid_argument);
  std::vector<uint8_t> data(2 * 144 - 1);
  std::vector<float> out(2 * 256);
  EXPECT_THROW(
      decode_gguf_kquant(
          "gguf:q4_k", data.data(), data.size(), 2, 256, out.data()),
      std::invalid_argument);
  EXPECT_THROW(
      decode_gguf_kquant("gguf:q4_k", data.data(), 0, -1, 256, out.data()),
      std::invalid_argument);
  // rows * 144 wraps to 144 bytes.
  EXPECT_THROW(
      decode_gguf_kquant(
          "gguf:q4_k",
          data.data(),
          144,
          (int64_t{1} << 62) + 1,
          256,
          out.data()),
      std::invalid_argument);
}

// Random blocks with finite fp16 scales, decoded by ggml's own dequantization.
TEST(ReferenceGgufKQuantTest, MatchesGgmlDequantization) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(-2.0f, 2.0f);
  const auto put_half = [&](uint8_t* bytes) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(scale(rng));
    std::memcpy(bytes, &h, sizeof(h));
  };
  for (const auto& [codec, type, block_bytes] :
       {std::tuple{"gguf:q4_k", GGML_TYPE_Q4_K, size_t{144}},
        std::tuple{"gguf:q6_k", GGML_TYPE_Q6_K, size_t{210}}}) {
    SCOPED_TRACE(codec);
    constexpr int64_t kRows = 5;
    constexpr int64_t kCols = 2 * kGgufKQuantBlockElements;
    std::vector<uint8_t> data(kRows * 2 * block_bytes);
    for (uint8_t& value : data) {
      value = static_cast<uint8_t>(byte(rng));
    }
    for (size_t block = 0; block < data.size(); block += block_bytes) {
      if (type == GGML_TYPE_Q4_K) {
        put_half(&data[block]);
        put_half(&data[block + 2]);
      } else {
        put_half(&data[block + 208]);
      }
    }
    std::vector<float> actual(kRows * kCols);
    decode_gguf_kquant(
        codec, data.data(), data.size(), kRows, kCols, actual.data());
    std::vector<float> expected(kRows * kCols);
    ggml_get_type_traits(type)->to_float(
        data.data(), expected.data(), kRows * kCols);
    EXPECT_EQ(
        std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)),
        0);
  }
}

// Every fp16 d, subnormals and infinities included, except NaNs, whose
// payload ggml's fp16 conversion may quiet.
TEST(ReferenceGgufKQuantTest, MatchesGgmlForEveryFp16Scale) {
  // Every weight stores u = 33 and every scale is 1, so each decodes to d.
  std::vector<uint8_t> block(128, 0x11); // ql: low nibbles 1
  block.resize(192, 0xAA); // qh: top bits 2
  block.resize(208, 1); // scales
  block.resize(210); // d
  const auto to_float = ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float;
  std::vector<float> expected(kGgufKQuantBlockElements);
  for (uint32_t h = 0; h <= 0xFFFF; ++h) {
    if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0) {
      continue;
    }
    block[208] = static_cast<uint8_t>(h);
    block[209] = static_cast<uint8_t>(h >> 8);
    to_float(block.data(), expected.data(), kGgufKQuantBlockElements);
    const std::vector<float> actual = decode_one_row("gguf:q6_k", block);
    ASSERT_EQ(
        std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)),
        0)
        << "fp16 d = " << h;
  }
}

TEST(ReferenceGgufKQuantTest, ThreadedDecodeMatchesSerialDecode) {
  // Just over four threads' worth of weights (2^20 each), in rows that do not
  // split evenly; the bytes need not be meaningful.
  constexpr int64_t kRows = 8193;
  constexpr int64_t kCols = 512;
  const size_t row_bytes = gguf_kquant_row_nbytes("gguf:q6_k", kCols);
  std::vector<uint8_t> data(kRows * row_bytes);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>((i * 2654435761u) >> 24);
  }
  std::vector<float> threaded(kRows * kCols);
  decode_gguf_kquant(
      "gguf:q6_k", data.data(), data.size(), kRows, kCols, threaded.data(), 4);
  std::vector<float> serial(kRows * kCols);
  decode_gguf_kquant(
      "gguf:q6_k", data.data(), data.size(), kRows, kCols, serial.data(), 1);
  EXPECT_EQ(
      std::memcmp(
          threaded.data(), serial.data(), threaded.size() * sizeof(float)),
      0);
}

} // namespace
} // namespace ptn::reference
