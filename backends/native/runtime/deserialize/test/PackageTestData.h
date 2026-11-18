// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace ptn::testing {

template <typename T>
void write_le(std::vector<uint8_t>& bytes, size_t offset, T value) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
  for (size_t i = 0; i < sizeof(T); ++i) {
    bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

inline uint32_t crc32(const uint8_t* data, size_t size) {
  uint32_t crc = std::numeric_limits<uint32_t>::max();
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

inline std::vector<uint8_t> make_safetensors(
    const std::string& header,
    size_t payload_size = 0) {
  std::vector<uint8_t> bytes(sizeof(uint64_t) + header.size() + payload_size);
  const uint64_t header_size = header.size();
  write_le(bytes, /*offset=*/0, header_size);
  std::memcpy(bytes.data() + sizeof(header_size), header.data(), header.size());
  return bytes;
}

struct StoredMember {
  std::string name;
  std::vector<uint8_t> payload;
};

inline std::vector<uint8_t> make_zip(const std::vector<StoredMember>& members) {
  constexpr uint32_t kEocdSignature = 0x06054b50;
  constexpr uint32_t kCentralDirectorySignature = 0x02014b50;
  constexpr uint32_t kLocalHeaderSignature = 0x04034b50;

  struct Entry {
    const StoredMember* member;
    uint32_t crc;
    uint32_t local_offset;
  };

  std::vector<uint8_t> bytes;
  std::vector<Entry> entries;
  for (const StoredMember& member : members) {
    const size_t offset = bytes.size();
    const size_t record_size = 30 + member.name.size() + member.payload.size();
    bytes.resize(offset + record_size);
    const uint32_t crc = crc32(member.payload.data(), member.payload.size());
    write_le(bytes, offset, kLocalHeaderSignature);
    write_le<uint16_t>(bytes, offset + 4, /*value=*/20);
    write_le(bytes, offset + 14, crc);
    write_le<uint32_t>(
        bytes, offset + 18, static_cast<uint32_t>(member.payload.size()));
    write_le<uint32_t>(
        bytes, offset + 22, static_cast<uint32_t>(member.payload.size()));
    write_le<uint16_t>(
        bytes, offset + 26, static_cast<uint16_t>(member.name.size()));
    std::memcpy(
        bytes.data() + offset + 30, member.name.data(), member.name.size());
    if (!member.payload.empty()) {
      std::memcpy(
          bytes.data() + offset + 30 + member.name.size(),
          member.payload.data(),
          member.payload.size());
    }
    entries.push_back(Entry{&member, crc, static_cast<uint32_t>(offset)});
  }

  const size_t central_offset = bytes.size();
  for (const Entry& entry : entries) {
    const size_t offset = bytes.size();
    bytes.resize(offset + 46 + entry.member->name.size());
    write_le(bytes, offset, kCentralDirectorySignature);
    write_le<uint16_t>(bytes, offset + 4, /*value=*/20);
    write_le<uint16_t>(bytes, offset + 6, /*value=*/20);
    write_le(bytes, offset + 16, entry.crc);
    write_le<uint32_t>(
        bytes,
        offset + 20,
        static_cast<uint32_t>(entry.member->payload.size()));
    write_le<uint32_t>(
        bytes,
        offset + 24,
        static_cast<uint32_t>(entry.member->payload.size()));
    write_le<uint16_t>(
        bytes, offset + 28, static_cast<uint16_t>(entry.member->name.size()));
    write_le(bytes, offset + 42, entry.local_offset);
    std::memcpy(
        bytes.data() + offset + 46,
        entry.member->name.data(),
        entry.member->name.size());
  }

  const size_t central_size = bytes.size() - central_offset;
  const size_t eocd = bytes.size();
  bytes.resize(eocd + 22);
  write_le(bytes, eocd, kEocdSignature);
  write_le<uint16_t>(bytes, eocd + 8, static_cast<uint16_t>(members.size()));
  write_le<uint16_t>(bytes, eocd + 10, static_cast<uint16_t>(members.size()));
  write_le<uint32_t>(bytes, eocd + 12, static_cast<uint32_t>(central_size));
  write_le<uint32_t>(bytes, eocd + 16, static_cast<uint32_t>(central_offset));
  return bytes;
}

} // namespace ptn::testing
