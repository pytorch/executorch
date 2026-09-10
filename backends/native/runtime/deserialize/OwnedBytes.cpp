// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ptn {
namespace {

std::string errno_suffix(int error) {
  return ": " + std::error_code(error, std::system_category()).message();
}

} // namespace

void OwnedBytes::Unmap::operator()(void* base) const noexcept {
#if !defined(_WIN32)
  // Nothing useful to do if this fails, and it must not throw: unique_ptr calls
  // this from its destructor.
  ::munmap(base, size);
#endif
}

OwnedBytes OwnedBytes::from_vector(std::vector<uint8_t> bytes) {
  return OwnedBytes(std::move(bytes));
}

OwnedBytes OwnedBytes::from_file(const std::string& path, bool use_mmap) {
  return use_mmap ? map_file(path) : read_file(path);
}

OwnedBytes OwnedBytes::read_file(const std::string& path) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::status(path, error);
  if (error) {
    throw std::runtime_error("cannot inspect " + path + ": " + error.message());
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw std::runtime_error("cannot read " + path + ": not a regular file");
  }
  const uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error("cannot size " + path + ": " + error.message());
  }
  if (file_size > std::numeric_limits<size_t>::max() ||
      file_size >
          static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("cannot read " + path + ": file is too large");
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  const size_t size = static_cast<size_t>(file_size);
  if (size == 0) {
    return OwnedBytes(std::vector<uint8_t>());
  }
  HeapBuffer buffer{std::make_unique_for_overwrite<uint8_t[]>(size), size};
  if (!file.read(
          reinterpret_cast<char*>(buffer.data.get()),
          static_cast<std::streamsize>(size))) {
    throw std::runtime_error("cannot read " + path);
  }
  return OwnedBytes(std::move(buffer));
}

OwnedBytes OwnedBytes::map_file(const std::string& path) {
#if defined(_WIN32)
  // TODO: Implement Windows mappings with CreateFileMapping and MapViewOfFile.
  throw std::runtime_error("cannot mmap " + path + ": unsupported platform");
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    const int error = errno;
    throw std::runtime_error("cannot open " + path + errno_suffix(error));
  }

  struct stat st = {};
  if (::fstat(fd, &st) < 0) {
    const std::string suffix = errno_suffix(errno);
    ::close(fd);
    throw std::runtime_error("cannot size " + path + suffix);
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    throw std::runtime_error("cannot mmap " + path + ": not a regular file");
  }
  if (st.st_size < 0) {
    ::close(fd);
    throw std::runtime_error("cannot mmap " + path + ": invalid file size");
  }
  if (static_cast<uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
    ::close(fd);
    throw std::runtime_error("cannot mmap " + path + ": file is too large");
  }
  const size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    return OwnedBytes(std::vector<uint8_t>());
  }

  // The whole file from offset 0, so the base is page-aligned and every span
  // into it has the same alignment it would have in a heap buffer. MAP_SHARED
  // lets other processes mapping this file share the same physical pages; the
  // mapping is read-only either way.
  void* base = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) {
    const int error = errno;
    ::close(fd);
    throw std::runtime_error("cannot mmap " + path + errno_suffix(error));
  }
  // The mapping keeps the file alive on its own, so the descriptor is dead
  // weight past this point.
  ::close(fd);
  return OwnedBytes(MappedFile(base, Unmap{size}));
#endif
}

ByteSpan OwnedBytes::span() const {
  if (const HeapBuffer* buffer = std::get_if<HeapBuffer>(&storage_)) {
    if (buffer->data == nullptr) {
      return {};
    }
    return ByteSpan(buffer->data.get(), buffer->size);
  }
  if (const MappedFile* mapped = std::get_if<MappedFile>(&storage_)) {
    if (mapped->get() == nullptr) {
      return {};
    }
    return ByteSpan(
        static_cast<const uint8_t*>(mapped->get()), mapped->get_deleter().size);
  }
  const std::vector<uint8_t>& bytes = std::get<std::vector<uint8_t>>(storage_);
  return ByteSpan(bytes.data(), bytes.size());
}

bool OwnedBytes::is_mapped() const {
  return std::holds_alternative<MappedFile>(storage_);
}

} // namespace ptn
