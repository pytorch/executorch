// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/utils/ReferenceGgufQuant.h>

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

constexpr int64_t kSuperBlockElements = 256;

std::vector<float> decode_one_row(
    std::string_view codec,
    const std::vector<uint8_t>& block,
    int64_t cols = kSuperBlockElements) {
  std::vector<float> out(cols);
  decode_gguf(codec, block.data(), block.size(), 1, cols, out.data());
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

// cppcheck-suppress syntaxError
TEST(ReferenceGgufQuantTest, DecodesQ4KBlock) {
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

TEST(ReferenceGgufQuantTest, DecodesQ5KBlock) {
  const std::array<uint8_t, 8> scales = {1, 2, 3, 4, 17, 5, 33, 63};
  const std::array<uint8_t, 8> mins = {0, 1, 2, 3, 4, 18, 5, 40};
  // Weight e stores the 5-bit value u, with scale pair e / 32. Its low four
  // bits are the low (even pair) or high (odd pair) nibble of qs byte
  // 32 * (e / 64) + e % 32, and its fifth bit is bit e / 32 of qh byte e % 32.
  // Each pair's weights hold different values, so reading the wrong pair fails.
  std::vector<uint8_t> qh(32);
  std::vector<uint8_t> qs(128);
  std::vector<float> expected(256);
  for (int e = 0; e < 256; ++e) {
    const int pair = e / 32;
    const int l = e % 32;
    const int u = (7 * l + 5 * pair) % 32;
    uint8_t& nibbles = qs[32 * (pair / 2) + l];
    nibbles |= static_cast<uint8_t>((u & 0xF) << (4 * (pair % 2)));
    qh[l] |= static_cast<uint8_t>((u >> 4) << pair);
    expected[e] = scales[pair] * u - 0.5f * mins[pair];
  }
  std::vector<uint8_t> block = {0x00, 0x3C, 0x00, 0x38}; // d = 1, dmin = 0.5
  const std::array<uint8_t, 12> packed = pack_q4_k_scales(scales, mins);
  block.insert(block.end(), packed.begin(), packed.end());
  block.insert(block.end(), qh.begin(), qh.end());
  block.insert(block.end(), qs.begin(), qs.end());
  ASSERT_EQ(block.size(), 176);

  EXPECT_EQ(decode_one_row("gguf:q5_k", block), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ6KBlock) {
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

TEST(ReferenceGgufQuantTest, DecodesQ8_0Block) {
  std::vector<uint8_t> block = {0x00, 0xB8}; // d = -0.5
  std::vector<float> expected;
  for (int j = 0; j < 32; ++j) {
    const int code = j == 31 ? 127 : 8 * j - 128;
    block.push_back(static_cast<uint8_t>(static_cast<int8_t>(code)));
    expected.push_back(-0.5f * code);
  }
  ASSERT_EQ(block.size(), 34);

  EXPECT_EQ(decode_one_row("gguf:q8_0", block, 32), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ3KBlock) {
  // Sixteen 6-bit scales s, each decoding as s - 32, packed as block_q3_K
  // stores them: the low nibbles of scales 0-7 in the low nibbles of bytes
  // 0-7 and those of scales 8-15 in their high nibbles, and the top two bits
  // of scale 4 k + i in bits 2 k and 2 k + 1 of byte 8 + i.
  std::array<int, 16> scales{};
  std::array<uint8_t, 12> packed{};
  for (int i = 0; i < 16; ++i) {
    scales[i] = (37 * i + 5) % 64;
    packed[i % 8] |= static_cast<uint8_t>((scales[i] & 0xF) << (4 * (i / 8)));
    packed[8 + i % 4] |=
        static_cast<uint8_t>((scales[i] >> 4) << (2 * (i / 4)));
  }
  // Weight e lies in half h = e / 128, plane p = e % 128 / 32 and position
  // l = e % 32: its 2-bit code is in bits 2 p of byte 32 h + l of the codes,
  // its high bit in bit 4 h + p of mask byte l, and it decodes to the code
  // minus 4 unless the high bit is set.
  std::vector<uint8_t> hmask(32);
  std::vector<uint8_t> qs(64);
  std::vector<float> expected(256);
  for (int e = 0; e < 256; ++e) {
    const int h = e / 128;
    const int p = e % 128 / 32;
    const int l = e % 32;
    const int code = (7 * e + h) % 4;
    const bool high = (5 * e + p) % 3 == 0;
    qs[32 * h + l] |= static_cast<uint8_t>(code << (2 * p));
    hmask[l] |= static_cast<uint8_t>(high << (4 * h + p));
    expected[e] = 0.5f * (scales[e / 16] - 32) * (code - (high ? 0 : 4));
  }
  std::vector<uint8_t> block = hmask;
  block.insert(block.end(), qs.begin(), qs.end());
  block.insert(block.end(), packed.begin(), packed.end());
  block.push_back(0x00); // d = 0.5
  block.push_back(0x38);
  ASSERT_EQ(block.size(), 110);

  EXPECT_EQ(decode_one_row("gguf:q3_k", block), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ2KBlock) {
  // Scale byte s holds the 4-bit scale (low nibble) and min (high nibble) of
  // weights 16 s to 16 s + 15. The 2-bit codes lie as in block_q3_K: weight e
  // of half h = e / 128, plane p = e % 128 / 32 and position l = e % 32 is in
  // bits 2 p of code byte 32 h + l.
  std::vector<uint8_t> block(84);
  for (int s = 0; s < 16; ++s) {
    block[s] = static_cast<uint8_t>((3 * s + 1) % 16 | ((5 * s + 2) % 16) << 4);
  }
  std::vector<float> expected(256);
  for (int e = 0; e < 256; ++e) {
    const int h = e / 128;
    const int p = e % 128 / 32;
    const int l = e % 32;
    const int code = (7 * e + h) % 4;
    block[16 + 32 * h + l] |= static_cast<uint8_t>(code << (2 * p));
    const uint8_t scale_min = block[e / 16];
    expected[e] = 0.5f * (scale_min & 0xF) * code - 0.25f * (scale_min >> 4);
  }
  block[81] = 0x38; // d = 0.5
  block[83] = 0x34; // dmin = 0.25

  EXPECT_EQ(decode_one_row("gguf:q2_k", block), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ4_0Block) {
  // Byte j's low nibble is weight j and its high nibble weight j + 16, each
  // stored as its code + 8.
  std::vector<uint8_t> block = {0x00, 0x34}; // d = 0.25
  std::vector<float> expected(32);
  for (int j = 0; j < 16; ++j) {
    block.push_back(static_cast<uint8_t>(j | (15 - j) << 4));
    expected[j] = 0.25f * (j - 8);
    expected[j + 16] = 0.25f * (7 - j);
  }
  ASSERT_EQ(block.size(), 18);

  EXPECT_EQ(decode_one_row("gguf:q4_0", block, 32), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ5_0Block) {
  // Weight j's 5-bit code keeps its low nibble in byte j % 16 of the codes,
  // low for j < 16 and high otherwise, and its fifth bit in bit j of qh.
  std::vector<uint8_t> qh(4);
  std::vector<uint8_t> qs(16);
  std::vector<float> expected(32);
  for (int j = 0; j < 32; ++j) {
    const int code = (7 * j + 3) % 32;
    qs[j % 16] |= static_cast<uint8_t>((code & 0xF) << (4 * (j / 16)));
    qh[j / 8] |= static_cast<uint8_t>((code >> 4) << (j % 8));
    expected[j] = -0.25f * (code - 16);
  }
  std::vector<uint8_t> block = {0x00, 0xB4}; // d = -0.25
  block.insert(block.end(), qh.begin(), qh.end());
  block.insert(block.end(), qs.begin(), qs.end());
  ASSERT_EQ(block.size(), 22);

  EXPECT_EQ(decode_one_row("gguf:q5_0", block, 32), expected);
}

TEST(ReferenceGgufQuantTest, DecodesQ5_1Block) {
  // The codes are laid out as in block_q5_0, each decoding to code * d + m.
  std::vector<uint8_t> qh(4);
  std::vector<uint8_t> qs(16);
  std::vector<float> expected(32);
  for (int j = 0; j < 32; ++j) {
    const int code = (5 * j + 1) % 32;
    qs[j % 16] |= static_cast<uint8_t>((code & 0xF) << (4 * (j / 16)));
    qh[j / 8] |= static_cast<uint8_t>((code >> 4) << (j % 8));
    expected[j] = 0.125f * code + 1.0f;
  }
  std::vector<uint8_t> block = {0x00, 0x30, 0x00, 0x3C}; // d = 0.125, m = 1
  block.insert(block.end(), qh.begin(), qh.end());
  block.insert(block.end(), qs.begin(), qs.end());
  ASSERT_EQ(block.size(), 24);

  EXPECT_EQ(decode_one_row("gguf:q5_1", block, 32), expected);
}

TEST(ReferenceGgufQuantTest, SizesWholeBlockRows) {
  EXPECT_TRUE(can_decode_gguf("gguf:q4_k"));
  EXPECT_TRUE(can_decode_gguf("gguf:q5_k"));
  EXPECT_TRUE(can_decode_gguf("gguf:q6_k"));
  EXPECT_TRUE(can_decode_gguf("gguf:q8_0"));
  EXPECT_TRUE(can_decode_gguf("gguf:q3_k"));
  EXPECT_TRUE(can_decode_gguf("gguf:q2_k"));
  EXPECT_TRUE(can_decode_gguf("gguf:q4_0"));
  EXPECT_TRUE(can_decode_gguf("gguf:q5_0"));
  EXPECT_TRUE(can_decode_gguf("gguf:q5_1"));
  EXPECT_FALSE(can_decode_gguf("gguf:iq4_xs"));
  EXPECT_FALSE(can_decode_gguf("q4_k"));
  EXPECT_EQ(gguf_row_nbytes("gguf:q4_k", 512), 288);
  EXPECT_EQ(gguf_row_nbytes("gguf:q5_k", 256), 176);
  EXPECT_EQ(gguf_row_nbytes("gguf:q6_k", 256), 210);
  EXPECT_EQ(gguf_row_nbytes("gguf:q8_0", 64), 68);
  EXPECT_THROW(gguf_row_nbytes("gguf:q8_0", 48), std::invalid_argument);
  EXPECT_EQ(gguf_row_nbytes("gguf:q3_k", 512), 220);
  EXPECT_EQ(gguf_row_nbytes("gguf:q2_k", 512), 168);
  EXPECT_EQ(gguf_row_nbytes("gguf:q4_0", 64), 36);
  EXPECT_THROW(gguf_row_nbytes("gguf:q4_0", 16), std::invalid_argument);
  EXPECT_EQ(gguf_row_nbytes("gguf:q5_0", 64), 44);
  EXPECT_THROW(gguf_row_nbytes("gguf:q5_0", 48), std::invalid_argument);
  EXPECT_EQ(gguf_row_nbytes("gguf:q5_1", 64), 48);
  EXPECT_THROW(gguf_row_nbytes("gguf:q5_1", 48), std::invalid_argument);
  EXPECT_THROW(gguf_row_nbytes("gguf:q4_k", 255), std::invalid_argument);
  EXPECT_THROW(gguf_row_nbytes("gguf:q4_k", 0), std::invalid_argument);
  EXPECT_THROW(gguf_row_nbytes("gguf:q4_k", -256), std::invalid_argument);
  EXPECT_THROW(gguf_row_nbytes("gguf:iq4_xs", 256), std::invalid_argument);
  std::vector<uint8_t> data(2 * 144 - 1);
  std::vector<float> out(2 * 256);
  EXPECT_THROW(
      decode_gguf("gguf:q4_k", data.data(), data.size(), 2, 256, out.data()),
      std::invalid_argument);
  EXPECT_THROW(
      decode_gguf("gguf:q4_k", data.data(), 0, -1, 256, out.data()),
      std::invalid_argument);
  // rows * 144 wraps to 144 bytes.
  EXPECT_THROW(
      decode_gguf(
          "gguf:q4_k",
          data.data(),
          144,
          (int64_t{1} << 62) + 1,
          256,
          out.data()),
      std::invalid_argument);
}

// Random blocks, with finite fp16 scales at their byte offsets in each block,
// decoded by ggml's own dequantization.
TEST(ReferenceGgufQuantTest, MatchesGgmlDequantization) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(-2.0f, 2.0f);
  const auto put_half = [&](uint8_t* bytes) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(scale(rng));
    std::memcpy(bytes, &h, sizeof(h));
  };
  for (const auto& [codec, type, scale_offsets] :
       {std::tuple{"gguf:q4_0", GGML_TYPE_Q4_0, std::vector<size_t>{0}},
        std::tuple{"gguf:q5_0", GGML_TYPE_Q5_0, std::vector<size_t>{0}},
        std::tuple{"gguf:q5_1", GGML_TYPE_Q5_1, std::vector<size_t>{0, 2}},
        std::tuple{"gguf:q8_0", GGML_TYPE_Q8_0, std::vector<size_t>{0}},
        std::tuple{"gguf:q2_k", GGML_TYPE_Q2_K, std::vector<size_t>{80, 82}},
        std::tuple{"gguf:q3_k", GGML_TYPE_Q3_K, std::vector<size_t>{108}},
        std::tuple{"gguf:q4_k", GGML_TYPE_Q4_K, std::vector<size_t>{0, 2}},
        std::tuple{"gguf:q5_k", GGML_TYPE_Q5_K, std::vector<size_t>{0, 2}},
        std::tuple{"gguf:q6_k", GGML_TYPE_Q6_K, std::vector<size_t>{208}}}) {
    SCOPED_TRACE(codec);
    constexpr int64_t kRows = 5;
    const int64_t cols = 2 * ggml_blck_size(type);
    const size_t block_bytes = ggml_type_size(type);
    std::vector<uint8_t> data(kRows * 2 * block_bytes);
    for (uint8_t& value : data) {
      value = static_cast<uint8_t>(byte(rng));
    }
    for (size_t block = 0; block < data.size(); block += block_bytes) {
      for (const size_t offset : scale_offsets) {
        put_half(&data[block + offset]);
      }
    }
    std::vector<float> actual(kRows * cols);
    decode_gguf(codec, data.data(), data.size(), kRows, cols, actual.data());
    std::vector<float> expected(kRows * cols);
    ggml_get_type_traits(type)->to_float(
        data.data(), expected.data(), kRows * cols);
    EXPECT_EQ(
        std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)),
        0);
  }
}

// Every fp16 d, subnormals and infinities included, except NaNs, whose
// payload ggml's fp16 conversion may quiet.
TEST(ReferenceGgufQuantTest, MatchesGgmlForEveryFp16Scale) {
  // Every weight stores u = 33 and every scale is 1, so each decodes to d.
  std::vector<uint8_t> block(128, 0x11); // ql: low nibbles 1
  block.resize(192, 0xAA); // qh: top bits 2
  block.resize(208, 1); // scales
  block.resize(210); // d
  const auto to_float = ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float;
  std::vector<float> expected(kSuperBlockElements);
  for (uint32_t h = 0; h <= 0xFFFF; ++h) {
    if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0) {
      continue;
    }
    block[208] = static_cast<uint8_t>(h);
    block[209] = static_cast<uint8_t>(h >> 8);
    to_float(block.data(), expected.data(), kSuperBlockElements);
    const std::vector<float> actual = decode_one_row("gguf:q6_k", block);
    ASSERT_EQ(
        std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)),
        0)
        << "fp16 d = " << h;
  }
}

TEST(ReferenceGgufQuantTest, ThreadedDecodeMatchesSerialDecode) {
  // Just over four threads' worth of weights (2^20 each), in rows that do not
  // split evenly; the bytes need not be meaningful.
  constexpr int64_t kRows = 8193;
  constexpr int64_t kCols = 512;
  const size_t row_bytes = gguf_row_nbytes("gguf:q6_k", kCols);
  std::vector<uint8_t> data(kRows * row_bytes);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>((i * 2654435761u) >> 24);
  }
  std::vector<float> threaded(kRows * kCols);
  decode_gguf(
      "gguf:q6_k", data.data(), data.size(), kRows, kCols, threaded.data(), 4);
  std::vector<float> serial(kRows * kCols);
  decode_gguf(
      "gguf:q6_k", data.data(), data.size(), kRows, kCols, serial.data(), 1);
  EXPECT_EQ(
      std::memcmp(
          threaded.data(), serial.data(), threaded.size() * sizeof(float)),
      0);
}

} // namespace
} // namespace ptn::reference
