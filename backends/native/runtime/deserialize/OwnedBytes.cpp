// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>

#include <cerrno>
#include <fstream>
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

std::string errno_suffix() {
  return ": " + std::error_code(errno, std::system_category()).message();
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
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  const std::streamsize size = file.tellg();
  if (size < 0) {
    throw std::runtime_error("cannot size " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("cannot read " + path);
  }
  return OwnedBytes(std::move(bytes));
}

OwnedBytes OwnedBytes::map_file(const std::string& path) {
#if defined(_WIN32)
  // TODO: Implement Windows mappings with CreateFileMapping and MapViewOfFile.
  throw std::runtime_error("cannot mmap " + path + ": unsupported platform");
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("cannot open " + path + errno_suffix());
  }

  struct stat st = {};
  if (::fstat(fd, &st) < 0) {
    const std::string suffix = errno_suffix();
    ::close(fd);
    throw std::runtime_error("cannot size " + path + suffix);
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    throw std::runtime_error("cannot mmap " + path + ": not a regular file");
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
  const std::string suffix = errno_suffix();
  // The mapping keeps the file alive on its own, so the descriptor is dead
  // weight past this point whether or not the map succeeded.
  ::close(fd);
  if (base == MAP_FAILED) {
    throw std::runtime_error("cannot mmap " + path + suffix);
  }
  return OwnedBytes(MappedFile(base, Unmap{size}));
#endif
}

ByteSpan OwnedBytes::span() const {
  if (const MappedFile* mapped = std::get_if<MappedFile>(&storage_)) {
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
