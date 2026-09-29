// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/ZipReader.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <zip.h>

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>

namespace ptn {
namespace {

struct ZipDeleter {
  void operator()(zip_t* archive) const noexcept {
    zip_discard(archive);
  }
};

struct ZipFileDeleter {
  void operator()(zip_file_t* file) const noexcept {
    zip_fclose(file);
  }
};

using ZipHandle = std::unique_ptr<zip_t, ZipDeleter>;
using ZipFileHandle = std::unique_ptr<zip_file_t, ZipFileDeleter>;

[[noreturn]] void throw_zip(zip_t* archive, const std::string& operation) {
  throw std::runtime_error("zip: " + operation + ": " + zip_strerror(archive));
}

[[noreturn]] void throw_zip_file(
    zip_file_t* file,
    const std::string& operation) {
  throw std::runtime_error(
      "zip: " + operation + ": " + zip_file_strerror(file));
}

constexpr uint32_t kLocalHeaderSignature = 0x04034b50;
constexpr size_t kLocalHeaderSize = 30;
constexpr uint16_t kDataDescriptorFlag = 1 << 3;

uint16_t read_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t read_u32(const uint8_t* p) {
  return static_cast<uint32_t>(read_u16(p)) |
      (static_cast<uint32_t>(read_u16(p + 2)) << 16);
}

ZipHandle open_path(const std::string& path) {
  int error_code = 0;
  ZipHandle archive(
      zip_open(path.c_str(), ZIP_RDONLY | ZIP_CHECKCONS, &error_code));
  if (archive == nullptr) {
    zip_error_t error;
    zip_error_init_with_code(&error, error_code);
    const std::string message =
        "zip: cannot open " + path + ": " + zip_error_strerror(&error);
    zip_error_fini(&error);
    throw std::runtime_error(message);
  }
  return archive;
}

ZipHandle open_memory(ByteSpan bytes) {
  zip_error_t error;
  zip_error_init(&error);
  zip_source_t* source =
      zip_source_buffer_create(bytes.data(), bytes.size(), 0, &error);
  if (source == nullptr) {
    const std::string message = "zip: cannot create memory source: " +
        std::string(zip_error_strerror(&error));
    zip_error_fini(&error);
    throw std::runtime_error(message);
  }

  ZipHandle archive(
      zip_open_from_source(source, ZIP_RDONLY | ZIP_CHECKCONS, &error));
  if (archive == nullptr) {
    zip_source_free(source);
    const std::string message = "zip: cannot open memory source: " +
        std::string(zip_error_strerror(&error));
    zip_error_fini(&error);
    throw std::runtime_error(message);
  }
  zip_error_fini(&error);
  return archive;
}

} // namespace

struct ZipReader::Impl {
  explicit Impl(ZipHandle handle, ByteSpan bytes = {})
      : archive(std::move(handle)), memory(bytes) {}

  ZipHandle archive;
  // The archive itself when memory-backed; empty when file-backed.
  ByteSpan memory;
  mutable std::mutex mutex;
};

ZipReader::ZipReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
  const zip_int64_t count = zip_get_num_entries(impl_->archive.get(), 0);
  if (count < 0) {
    throw_zip(impl_->archive.get(), "cannot enumerate members");
  }
  if (static_cast<uint64_t>(count) > detail::kMaxPackageMembers) {
    throw ResourceLimitError("zip: member count exceeds package limit");
  }

  names_.reserve(static_cast<size_t>(count));
  for (zip_uint64_t index = 0; index < static_cast<zip_uint64_t>(count);
       ++index) {
    zip_stat_t stat;
    zip_stat_init(&stat);
    if (zip_stat_index(impl_->archive.get(), index, ZIP_FL_UNCHANGED, &stat) !=
        0) {
      throw_zip(impl_->archive.get(), "cannot stat member");
    }
    constexpr zip_uint64_t kRequired = ZIP_STAT_NAME | ZIP_STAT_SIZE |
        ZIP_STAT_COMP_METHOD | ZIP_STAT_ENCRYPTION_METHOD;
    if ((stat.valid & kRequired) != kRequired || stat.name == nullptr) {
      throw std::runtime_error("zip: member metadata is incomplete");
    }
    const std::string name(stat.name);
    if (stat.comp_method != ZIP_CM_STORE) {
      throw std::runtime_error("zip: member is compressed: " + name);
    }
    if (stat.encryption_method != ZIP_EM_NONE) {
      throw std::runtime_error("zip: member is encrypted: " + name);
    }
    if (stat.size > detail::kMaxPackageBytes) {
      throw ResourceLimitError(
          "zip: member exceeds package size limit: " + name);
    }
    if (stat.size > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("zip: member is too large: " + name);
    }
    if (!entries_.emplace(name, Entry{index, static_cast<size_t>(stat.size)})
             .second) {
      throw std::runtime_error("zip: duplicate member name: " + name);
    }
    names_.push_back(name);
  }
  if (!impl_->memory.empty()) {
    locate_member_data();
  }
}

// libzip does not expose where member data starts, so walk the local headers.
// Accepts only the layout zip writers produce by default, one stored member
// after another in central-directory order, and otherwise leaves every offset
// unset so reads go through libzip.
void ZipReader::locate_member_data() {
  const ByteSpan archive = impl_->memory;
  std::vector<size_t> offsets;
  offsets.reserve(names_.size());
  size_t position = 0;
  for (const std::string& name : names_) {
    if (archive.size() - position < kLocalHeaderSize) {
      return;
    }
    const uint8_t* header = archive.data() + position;
    const size_t name_size = read_u16(header + 26);
    const size_t extra_size = read_u16(header + 28);
    if (read_u32(header) != kLocalHeaderSignature ||
        (read_u16(header + 6) & kDataDescriptorFlag) != 0 ||
        read_u16(header + 8) != ZIP_CM_STORE) {
      return;
    }
    const size_t name_start = position + kLocalHeaderSize;
    if (archive.size() - name_start < name_size + extra_size ||
        std::string_view(
            reinterpret_cast<const char*>(archive.data() + name_start),
            name_size) != name) {
      return;
    }
    const size_t data = name_start + name_size + extra_size;
    const size_t size = entries_.find(name)->second.size;
    if (archive.size() - data < size) {
      return;
    }
    offsets.push_back(data);
    position = data + size;
  }
  for (size_t i = 0; i < names_.size(); ++i) {
    entries_.find(names_[i])->second.data_offset = offsets[i];
  }
}

ZipReader::~ZipReader() = default;
ZipReader::ZipReader(ZipReader&&) noexcept = default;
ZipReader& ZipReader::operator=(ZipReader&&) noexcept = default;

ZipReader ZipReader::open(const std::string& path) {
  return ZipReader(std::make_unique<Impl>(open_path(path)));
}

ZipReader ZipReader::open(ByteSpan archive) {
  return ZipReader(std::make_unique<Impl>(open_memory(archive), archive));
}

std::optional<size_t> ZipReader::member_size(std::string_view name) const {
  const Entry* entry = find_entry(name);
  return entry == nullptr ? std::nullopt : std::optional<size_t>(entry->size);
}

std::optional<ByteSpan> ZipReader::member_bytes(std::string_view name) const {
  const Entry* entry = find_entry(name);
  if (entry == nullptr || !entry->data_offset) {
    return std::nullopt;
  }
  return impl_->memory.subspan(*entry->data_offset, entry->size);
}

std::vector<uint8_t> ZipReader::read(std::string_view name) const {
  const Entry* entry = find_entry(name);
  if (entry == nullptr) {
    throw std::runtime_error("zip: no member named " + std::string(name));
  }
  std::vector<uint8_t> bytes(entry->size);
  read_entry_into(name, *entry, 0, MutableByteSpan(bytes));
  return bytes;
}

void ZipReader::read_into(
    std::string_view name,
    size_t offset,
    MutableByteSpan destination) const {
  const Entry* entry = find_entry(name);
  if (entry == nullptr) {
    throw std::runtime_error("zip: no member named " + std::string(name));
  }
  read_entry_into(name, *entry, offset, destination);
}

const ZipReader::Entry* ZipReader::find_entry(std::string_view name) const {
  const auto entry = entries_.find(name);
  return entry == entries_.end() ? nullptr : &entry->second;
}

void ZipReader::read_entry_into(
    std::string_view name,
    const Entry& entry,
    size_t offset,
    MutableByteSpan destination) const {
  if (offset > entry.size || destination.size() > entry.size - offset) {
    throw std::runtime_error(
        "zip: read range is outside member " + std::string(name));
  }
  if (destination.empty()) {
    return;
  }
  if (entry.data_offset) {
    std::memcpy(
        destination.data(),
        impl_->memory.data() + *entry.data_offset + offset,
        destination.size());
    return;
  }

  const std::lock_guard<std::mutex> lock(impl_->mutex);
  ZipFileHandle file(
      zip_fopen_index(impl_->archive.get(), entry.index, ZIP_FL_UNCHANGED));
  if (file == nullptr) {
    throw_zip(impl_->archive.get(), "cannot open member " + std::string(name));
  }
  if (zip_fseek(file.get(), static_cast<zip_int64_t>(offset), SEEK_SET) != 0) {
    throw_zip_file(file.get(), "cannot seek member " + std::string(name));
  }

  size_t written = 0;
  while (written < destination.size()) {
    const zip_uint64_t request =
        static_cast<zip_uint64_t>(destination.size() - written);
    const zip_int64_t count =
        zip_fread(file.get(), destination.data() + written, request);
    if (count < 0) {
      throw_zip_file(file.get(), "cannot read member " + std::string(name));
    }
    if (count == 0) {
      throw std::runtime_error(
          "zip: unexpected end of member " + std::string(name));
    }
    written += static_cast<size_t>(count);
  }
}

void ZipReader::verify(std::string_view name) const {
  const Entry* entry = find_entry(name);
  if (entry == nullptr) {
    throw std::runtime_error("zip: no member named " + std::string(name));
  }

  const std::lock_guard<std::mutex> lock(impl_->mutex);
  ZipFileHandle file(
      zip_fopen_index(impl_->archive.get(), entry->index, ZIP_FL_UNCHANGED));
  if (file == nullptr) {
    throw_zip(impl_->archive.get(), "cannot open member " + std::string(name));
  }

  std::array<uint8_t, 64 * 1024> buffer{};
  size_t read = 0;
  while (read < entry->size) {
    const size_t request = std::min(buffer.size(), entry->size - read);
    const zip_int64_t count = zip_fread(file.get(), buffer.data(), request);
    if (count < 0) {
      throw_zip_file(file.get(), "cannot verify member " + std::string(name));
    }
    if (count == 0) {
      throw std::runtime_error(
          "zip: unexpected end of member " + std::string(name));
    }
    read += static_cast<size_t>(count);
  }

  uint8_t extra = 0;
  const zip_int64_t count = zip_fread(file.get(), &extra, 1);
  if (count < 0) {
    throw_zip_file(file.get(), "cannot verify member " + std::string(name));
  }
  if (count != 0) {
    throw std::runtime_error(
        "zip: member is larger than its metadata: " + std::string(name));
  }
}

} // namespace ptn
