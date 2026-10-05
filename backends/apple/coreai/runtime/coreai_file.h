/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unistd.h>
#include <cerrno>

namespace executorch::backends::coreai {

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor() {
    reset(-1);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
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

} // namespace executorch::backends::coreai
