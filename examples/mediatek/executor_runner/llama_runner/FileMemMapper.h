/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <executorch/runtime/platform/log.h>

#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace example {

class FileMemMapper { // Read-only mmap
 public:
  explicit FileMemMapper(const std::string& path) {
    // Get fd
    mFd = open(path.c_str(), O_RDONLY);
    if (mFd == -1) {
      ET_LOG(Error, "Open file fail: %s", path.c_str());
      return;
    }

    // Get size
    struct stat sb;
    if (fstat(mFd, &sb) == -1) {
      ET_LOG(Error, "fstat fail");
      return;
    }
    mSize = sb.st_size;

    // Map file data to memory
    mBuffer = mmap(NULL, mSize, PROT_READ, MAP_SHARED, mFd, 0);
    if (mBuffer == MAP_FAILED) {
      ET_LOG(Error, "mmap fail");
      return;
    }

    ET_LOG(
        Debug,
        "FileMemMapper: Mapped to (fd=%d, size=%zu, addr=%p): %s",
        mFd,
        mSize,
        mBuffer,
        path.c_str());
  }

  // Move ctor
  explicit FileMemMapper(FileMemMapper&& other)
      : mFd(other.mFd), mBuffer(other.mBuffer), mSize(other.mSize) {
    other.mFd = -1;
    other.mBuffer = nullptr;
    other.mSize = 0;
  }

  ~FileMemMapper() {
    if (!mBuffer && mFd == -1) {
      return;
    }

    ET_LOG(
        Debug,
        "FileMemMapper: Unmapping (fd=%d, size=%zu, addr=%p)",
        mFd,
        mSize,
        mBuffer);

    if (mBuffer && munmap(mBuffer, mSize) == -1) {
      ET_LOG(Error, "munmap fail");
      return;
    }

    if (mFd != -1 && close(mFd) == -1) {
      ET_LOG(Error, "close fail");
      return;
    }
  }

  template <typename T = void*>
  T getAddr() const {
    return reinterpret_cast<T>(mBuffer);
  }

  size_t getSize() const {
    return mSize;
  }

 private:
  int mFd = -1;
  void* mBuffer = nullptr;
  size_t mSize = 0;
};

} // namespace example
