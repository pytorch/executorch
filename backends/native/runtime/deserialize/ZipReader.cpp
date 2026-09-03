// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/ZipReader.h>

#include <bit>
#include <concepts>
#include <cstring>
#include <stdexcept>

namespace ptn {
namespace {

// Record signatures, as they read back from disk little-endian.
constexpr uint32_t kEocdSig = 0x06054b50; // PK\5\6
constexpr uint32_t kZip64EocdSig = 0x06064b50; // PK\6\6
constexpr uint32_t kZip64LocatorSig = 0x07064b50; // PK\6\7
constexpr uint32_t kCentralDirSig = 0x02014b50; // PK\1\2
constexpr uint32_t kLocalHeaderSig = 0x04034b50; // PK\3\4

constexpr size_t kEocdSize = 22;
constexpr size_t kZip64LocatorSize = 20;
constexpr size_t kZip64EocdSize = 56;
constexpr size_t kCentralDirEntrySize = 46;
constexpr size_t kLocalHeaderSize = 30;

// A 32-bit size/offset field holding this sentinel means the real 64-bit value
// lives in the entry's zip64 extra field.
constexpr uint32_t kZip64Sentinel = 0xffffffffu;
constexpr uint16_t kZip64ExtraId = 0x0001;
constexpr uint16_t kMethodStored = 0;
constexpr uint16_t kFlagEncrypted = 0x0001;
// Upper bound on the archive comment trailing the EOCD record, hence on how far
// back the record can sit.
constexpr size_t kMaxCommentSize = 0xffff;

void require(bool ok, const char* what) {
  if (!ok) {
    throw std::runtime_error(std::string("zip: ") + what);
  }
}

// Every field in a zip record is a little-endian unsigned integer, so one
// bounds-checked load serves all three widths.
template <std::unsigned_integral T>
T read_le(ByteSpan s, size_t off) {
  static_assert(
      std::endian::native == std::endian::little,
      "zip fields are little-endian; a big-endian host needs a byte swap here");
  require(off + sizeof(T) <= s.size(), "read past end of buffer");
  T value = 0;
  std::memcpy(&value, s.data() + off, sizeof(T));
  return value;
}

uint16_t u16(ByteSpan s, size_t off) {
  return read_le<uint16_t>(s, off);
}

uint32_t u32(ByteSpan s, size_t off) {
  return read_le<uint32_t>(s, off);
}

uint64_t u64(ByteSpan s, size_t off) {
  return read_le<uint64_t>(s, off);
}

// Offset of the end-of-central-directory record. Scanned backwards because the
// record is followed by a variable-length comment.
size_t find_eocd(ByteSpan archive) {
  require(
      archive.size() >= kEocdSize,
      "buffer is smaller than an end-of-central-directory record");
  const size_t highest = archive.size() - kEocdSize;
  const size_t lowest = archive.size() > kEocdSize + kMaxCommentSize
      ? archive.size() - kEocdSize - kMaxCommentSize
      : 0;
  for (size_t pos = highest + 1; pos > lowest;) {
    --pos;
    if (u32(archive, pos) != kEocdSig) {
      continue;
    }
    // The signature can also occur inside the comment. Accept this candidate
    // only if its own comment-length field accounts for exactly the bytes that
    // follow the record.
    if (u16(archive, pos + 20) == archive.size() - pos - kEocdSize) {
      return pos;
    }
  }
  throw std::runtime_error("zip: no end-of-central-directory record");
}

struct CentralDirInfo {
  uint64_t count = 0;
  uint64_t size = 0;
  uint64_t offset = 0;
};

// Central-directory location, preferring the zip64 record when the classic one
// is backed by a zip64 locator.
CentralDirInfo read_central_dir_info(ByteSpan archive, size_t eocd) {
  CentralDirInfo info;
  info.count = u16(archive, eocd + 10);
  info.size = u32(archive, eocd + 12);
  info.offset = u32(archive, eocd + 16);

  if (eocd < kZip64LocatorSize) {
    return info;
  }
  const size_t locator = eocd - kZip64LocatorSize;
  if (u32(archive, locator) != kZip64LocatorSig) {
    return info;
  }

  const uint64_t record = u64(archive, locator + 8);
  require(
      record <= archive.size() && archive.size() - record >= kZip64EocdSize,
      "zip64 end-of-central-directory offset out of range");
  const size_t z64 = static_cast<size_t>(record);
  require(
      u32(archive, z64) == kZip64EocdSig,
      "bad zip64 end-of-central-directory signature");
  info.count = u64(archive, z64 + 32);
  info.size = u64(archive, z64 + 40);
  info.offset = u64(archive, z64 + 48);
  return info;
}

// Consume the next 64-bit value from a zip64 extra field's payload.
uint64_t take_u64(ByteSpan field, size_t& cursor) {
  require(cursor + 8 <= field.size(), "zip64 extra field is too short");
  const uint64_t value = u64(field, cursor);
  cursor += 8;
  return value;
}

// Replace whichever of the three fields held the zip64 sentinel with its real
// value. The extra field stores them in a fixed order but includes only the
// ones that were sentinelled, so they must be consumed conditionally.
void apply_zip64_extra(
    ByteSpan extra,
    bool want_usize,
    bool want_csize,
    bool want_offset,
    uint64_t& usize,
    uint64_t& csize,
    uint64_t& offset) {
  size_t pos = 0;
  while (pos + 4 <= extra.size()) {
    const uint16_t id = u16(extra, pos);
    const uint16_t len = u16(extra, pos + 2);
    require(pos + 4 + len <= extra.size(), "extra field runs past its record");
    if (id == kZip64ExtraId) {
      const ByteSpan field = extra.subspan(pos + 4, len);
      size_t cursor = 0;
      if (want_usize) {
        usize = take_u64(field, cursor);
      }
      if (want_csize) {
        csize = take_u64(field, cursor);
      }
      if (want_offset) {
        offset = take_u64(field, cursor);
      }
      return;
    }
    pos += 4 + len;
  }
  throw std::runtime_error("zip: entry needs a zip64 extra field but has none");
}

// Where a member's payload starts. Only the local header's name/extra lengths
// are trusted; its size fields may be zeroed in favour of a data descriptor.
size_t payload_offset(ByteSpan archive, uint64_t local_header) {
  require(
      local_header <= archive.size() &&
          archive.size() - local_header >= kLocalHeaderSize,
      "local file header offset out of range");
  const size_t lh = static_cast<size_t>(local_header);
  require(
      u32(archive, lh) == kLocalHeaderSig, "bad local file header signature");
  return lh + kLocalHeaderSize + u16(archive, lh + 26) + u16(archive, lh + 28);
}

} // namespace

ZipReader ZipReader::open(ByteSpan archive) {
  const size_t eocd = find_eocd(archive);
  const CentralDirInfo dir = read_central_dir_info(archive, eocd);
  require(
      dir.offset <= archive.size() && archive.size() - dir.offset >= dir.size,
      "central directory out of range");

  ZipReader out;
  size_t pos = static_cast<size_t>(dir.offset);
  const size_t dir_end = pos + static_cast<size_t>(dir.size);

  for (uint64_t i = 0; i < dir.count; ++i) {
    require(
        pos + kCentralDirEntrySize <= dir_end, "central directory truncated");
    require(
        u32(archive, pos) == kCentralDirSig,
        "bad central-directory entry signature");

    const uint16_t flags = u16(archive, pos + 8);
    const uint16_t method = u16(archive, pos + 10);
    uint64_t csize = u32(archive, pos + 20);
    uint64_t usize = u32(archive, pos + 24);
    const uint16_t name_len = u16(archive, pos + 28);
    const uint16_t extra_len = u16(archive, pos + 30);
    const uint16_t comment_len = u16(archive, pos + 32);
    uint64_t local_header = u32(archive, pos + 42);

    const size_t name_at = pos + kCentralDirEntrySize;
    const size_t extra_at = name_at + name_len;
    const size_t comment_at = extra_at + extra_len;
    require(
        comment_at + comment_len <= dir_end,
        "central-directory entry truncated");

    const std::string name(
        reinterpret_cast<const char*>(archive.data() + name_at), name_len);

    if (usize == kZip64Sentinel || csize == kZip64Sentinel ||
        local_header == kZip64Sentinel) {
      apply_zip64_extra(
          archive.subspan(extra_at, extra_len),
          usize == kZip64Sentinel,
          csize == kZip64Sentinel,
          local_header == kZip64Sentinel,
          usize,
          csize,
          local_header);
    }

    if ((flags & kFlagEncrypted) != 0) {
      throw std::runtime_error("zip: member is encrypted: " + name);
    }
    if (method != kMethodStored) {
      throw std::runtime_error(
          "zip: member is compressed, only stored members are supported: " +
          name);
    }
    if (csize != usize) {
      throw std::runtime_error(
          "zip: stored member with mismatched sizes: " + name);
    }

    const size_t payload = payload_offset(archive, local_header);
    if (payload > archive.size() || archive.size() - payload < usize) {
      throw std::runtime_error(
          "zip: member payload runs past the end of the archive: " + name);
    }

    if (!out.members_.emplace(name, archive.subspan(payload, usize)).second) {
      throw std::runtime_error("zip: duplicate member name: " + name);
    }
    out.names_.push_back(name);
    pos = comment_at + comment_len;
  }

  return out;
}

std::optional<ByteSpan> ZipReader::member(const std::string& name) const {
  const auto it = members_.find(name);
  if (it == members_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace ptn
