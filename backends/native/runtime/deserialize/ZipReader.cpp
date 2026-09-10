// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/ZipReader.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

#include <zip.h>

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
  explicit Impl(ZipHandle handle) : archive(std::move(handle)) {}

  ZipHandle archive;
};

ZipReader::ZipReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
  const zip_int64_t count = zip_get_num_entries(impl_->archive.get(), 0);
  if (count < 0) {
    throw_zip(impl_->archive.get(), "cannot enumerate members");
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
    if (stat.size > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("zip: member is too large: " + name);
    }
    if (!entries_.emplace(name, Entry{index, static_cast<size_t>(stat.size)})
             .second) {
      throw std::runtime_error("zip: duplicate member name: " + name);
    }
    names_.push_back(name);
  }
}

ZipReader::~ZipReader() = default;
ZipReader::ZipReader(ZipReader&&) noexcept = default;
ZipReader& ZipReader::operator=(ZipReader&&) noexcept = default;

ZipReader ZipReader::open(const std::string& path) {
  return ZipReader(std::make_unique<Impl>(open_path(path)));
}

ZipReader ZipReader::open(ByteSpan archive) {
  return ZipReader(std::make_unique<Impl>(open_memory(archive)));
}

std::optional<size_t> ZipReader::member_size(std::string_view name) const {
  const Entry* entry = find_entry(name);
  return entry == nullptr ? std::nullopt : std::optional<size_t>(entry->size);
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

} // namespace ptn
