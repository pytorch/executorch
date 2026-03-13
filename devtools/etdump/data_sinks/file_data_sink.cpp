/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/data_sinks/file_data_sink.h>
#include <cstdio> // For FILE operations

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace executorch {
namespace etdump {

FileDataSink::FileDataSink(FileDataSink&& other) noexcept
    : file_(other.file_), total_written_bytes_(other.total_written_bytes_) {
  other.file_ = nullptr;
}

Result<FileDataSink> FileDataSink::create(const char* file_path) {
  // Open the file and get the file pointer
  FILE* file = fopen(file_path, "wb");
  if (!file) {
    // Return an error if the file cannot be accessed or created
    ET_LOG(Error, "File %s cannot be accessed or created.", file_path);
    return Error::AccessFailed;
  }

  // Return the successfully created FileDataSink
  return FileDataSink(file);
}

FileDataSink::~FileDataSink() {
  // Close the file
  close();
}

Result<size_t> FileDataSink::write(const void* ptr, size_t size) {
  if (!file_) {
    ET_LOG(Error, "File not open, unable to write.");
    return Error::AccessFailed;
  }

  bool inPlaceTensor = false;

  if (size != 0 && ptr == nullptr) {
    inPlaceTensor = true;
  } else if (size == 0 || ptr == nullptr) {
    ET_LOG(Info, "Invalid data to write to file");
    return total_written_bytes_;
  }

  size_t offset = total_written_bytes_;

  if (inPlaceTensor) {
    std::vector<uint8_t> zeros(size, 0);
    size_t written = fwrite(zeros.data(), 1, size, file_);
    if (written != size) {
      ET_LOG(Error, "Write failed: wrote %zu bytes of %zu", written, size);
      return Error::Internal;
    }
    total_written_bytes_ += written;
  } else {
    size_t written = fwrite(ptr, 1, size, file_);
    if (written != size) {
      ET_LOG(Error, "Write failed: wrote %zu bytes of %zu", written, size);
      return Error::Internal;
    }
    total_written_bytes_ += written;
  }

  return offset;
}

size_t FileDataSink::get_used_bytes() const {
  return total_written_bytes_;
}

void FileDataSink::close() {
  if (file_) {
    fclose(file_);
    file_ = nullptr;
  }
}

} // namespace etdump
} // namespace executorch
