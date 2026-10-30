/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

namespace executorch::backends::coreai {

enum class StorageCreationMode {
  OpenExisting,
  Create,
};

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd = -1) : fd_(fd) {}
  ~FileDescriptor() {
    reset(-1);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.release()) {}
  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }
  int get() const {
    return fd_;
  }
  int release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void reset(int fd) {
    if (fd_ >= 0) {
      // Retrying close can close a descriptor reused by another thread.
      ::close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_;
};

template <typename F>
auto retry_eintr(F call) {
  decltype(call()) result;
  do {
    result = call();
  } while (result == -1 && errno == EINTR);
  return result;
}

// Opens a child directory without following symlinks. Returns -1 with errno
// set, including ENOENT when the child is missing.
inline int open_directory_at(int parent_fd, const char* name) {
  return retry_eintr([&] {
    return openat(
        parent_fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  });
}

// Reads type and permission metadata for a child without following symlinks.
inline int lstat_at(int parent_fd, const char* name, struct stat* info) {
  return retry_eintr(
      [&] { return fstatat(parent_fd, name, info, AT_SYMLINK_NOFOLLOW); });
}

} // namespace executorch::backends::coreai
