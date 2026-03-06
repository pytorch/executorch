/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_cache/file_backend_cache.h>

#ifdef _WIN32
#include <direct.h>
#endif
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

namespace executorch {
namespace extension {

namespace {

bool mkdirs(const std::string& path) {
  if (path.empty()) {
    return true;
  }
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  auto pos = path.find_last_of('/');
  if (pos != std::string::npos && pos > 0) {
    if (!mkdirs(path.substr(0, pos))) {
      return false;
    }
  }
#ifdef _WIN32
  return _mkdir(path.c_str()) == 0;
#else
  return mkdir(path.c_str(), 0755) == 0;
#endif
}

void free_aligned_buffer(void* context, void* data, size_t /*size*/) {
  auto alignment =
      static_cast<std::align_val_t>(reinterpret_cast<uintptr_t>(context));
  ::operator delete(data, alignment);
}

} // namespace

FileBackendCache::FileBackendCache(std::string cache_dir)
    : cache_dir_(std::move(cache_dir)) {}

std::string FileBackendCache::key_to_path(
    const char* backend_id,
    size_t delegate_index,
    const char* key) const {
  return cache_dir_ + "/" + backend_id + "/" + std::to_string(delegate_index) +
      "/" + key;
}

runtime::Result<runtime::FreeableBuffer> FileBackendCache::load(
    const char* backend_id,
    size_t delegate_index,
    const char* key,
    size_t alignment) const {
  std::string path = key_to_path(backend_id, delegate_index, key);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return runtime::Error::NotFound;
  }

  std::fseek(f, 0, SEEK_END);
  long file_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);

  if (file_size <= 0) {
    std::fclose(f);
    return runtime::Error::NotFound;
  }

  void* buf = ::operator new(
      static_cast<size_t>(file_size),
      static_cast<std::align_val_t>(alignment),
      std::nothrow);
  if (!buf) {
    std::fclose(f);
    return runtime::Error::MemoryAllocationFailed;
  }

  size_t read = std::fread(buf, 1, static_cast<size_t>(file_size), f);
  std::fclose(f);

  if (read != static_cast<size_t>(file_size)) {
    ::operator delete(buf, static_cast<std::align_val_t>(alignment));
    return runtime::Error::AccessFailed;
  }

  return runtime::FreeableBuffer(
      buf,
      static_cast<size_t>(file_size),
      free_aligned_buffer,
      reinterpret_cast<void*>(alignment));
}

runtime::Error FileBackendCache::save(
    const char* backend_id,
    size_t delegate_index,
    const char* key,
    const void* data,
    size_t size) {
  std::string path = key_to_path(backend_id, delegate_index, key);

  auto pos = path.find_last_of('/');
  if (pos != std::string::npos) {
    if (!mkdirs(path.substr(0, pos))) {
      return runtime::Error::AccessFailed;
    }
  }

  std::string tmp_path = path + ".tmp";
  FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) {
    return runtime::Error::AccessFailed;
  }

  size_t written = std::fwrite(data, 1, size, f);
  std::fclose(f);

  if (written != size) {
    std::remove(tmp_path.c_str());
    return runtime::Error::AccessFailed;
  }

  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    return runtime::Error::AccessFailed;
  }

  return runtime::Error::Ok;
}

runtime::Error FileBackendCache::remove(
    const char* backend_id,
    size_t delegate_index,
    const char* key) {
  std::string path = key_to_path(backend_id, delegate_index, key);
  if (std::remove(path.c_str()) != 0) {
    if (errno == ENOENT) {
      return runtime::Error::NotFound;
    }
    return runtime::Error::AccessFailed;
  }
  return runtime::Error::Ok;
}

} // namespace extension
} // namespace executorch
