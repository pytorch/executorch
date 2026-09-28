/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_data/file_data_writer.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compat_unistd.h>
#include <executorch/runtime/platform/log.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN64)
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

using executorch::runtime::Error;
using executorch::runtime::Span;

namespace executorch {
namespace extension {
namespace {

bool add_overflows(size_t lhs, size_t rhs, size_t* result) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return true;
  }
  *result = lhs + rhs;
  return false;
}

std::string parent_directory(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  if (slash == std::string::npos) {
    return ".";
  }
  if (slash == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, slash);
}

std::string base_name(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

Error log_io_error(const char* operation, const std::string& path) {
  ET_LOG(
      Error,
      "%s failed for %s: %s (%d)",
      operation,
      path.c_str(),
      std::strerror(errno),
      errno);
  return Error::AccessFailed;
}

#if defined(_WIN64)
ssize_t write_at(int fd, const void* data, size_t size, size_t offset) {
  if (::_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0) {
    return -1;
  }
  return ::_write(fd, data, static_cast<unsigned int>(size));
}
#else
ssize_t write_at(int fd, const void* data, size_t size, size_t offset) {
  return ::pwrite(fd, data, size, static_cast<off_t>(offset));
}
#endif

Error write_all(
    int fd,
    const uint8_t* data,
    size_t size,
    size_t offset,
    const std::string& path) {
  while (size > 0) {
    const size_t chunk = std::min(
        size, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const ssize_t written = write_at(fd, data, chunk, offset);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return log_io_error("write", path);
    }
    data += written;
    size -= static_cast<size_t>(written);
    offset += static_cast<size_t>(written);
  }
  return Error::Ok;
}

Error resize_file(int fd, size_t size, const std::string& path) {
#if defined(_WIN64)
  if (::_chsize_s(fd, size) != 0) {
#else
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
#endif
    return log_io_error("resize", path);
  }
  return Error::Ok;
}

Error sync_file(int fd, const std::string& path) {
#if defined(_WIN64)
  if (::_commit(fd) != 0) {
#else
  if (::fsync(fd) != 0) {
#endif
    return log_io_error("sync", path);
  }
  return Error::Ok;
}

Error publish_file(
    const std::string& temporary_path,
    const std::string& destination_path,
    bool* replaced) {
  *replaced = false;
#if defined(_WIN64)
  if (!::MoveFileExA(
          temporary_path.c_str(),
          destination_path.c_str(),
          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    errno = EACCES;
    return log_io_error("replace", destination_path);
  }
  *replaced = true;
  return Error::Ok;
#else
  if (::rename(temporary_path.c_str(), destination_path.c_str()) != 0) {
    return log_io_error("rename", destination_path);
  }
  *replaced = true;

  const std::string directory = parent_directory(destination_path);
  const int directory_fd = ::open(directory.c_str(), O_RDONLY);
  if (directory_fd < 0) {
    return log_io_error("open directory", directory);
  }
  const int sync_result = ::fsync(directory_fd);
  const int sync_errno = errno;
  ::close(directory_fd);
  if (sync_result != 0) {
    errno = sync_errno;
    return log_io_error("sync directory", directory);
  }
  return Error::Ok;
#endif
}

} // namespace

class FileDataWriterState final {
 public:
  FileDataWriterState(int fd, std::string temporary_path)
      : fd_(fd), temporary_path_(std::move(temporary_path)) {}

  ~FileDataWriterState() {
    discard();
  }

  Error write(Span<const uint8_t> data, size_t offset) {
    ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Writer is not active");
    size_t end;
    ET_CHECK_OR_RETURN_ERROR(
        !add_overflows(offset, data.size(), &end),
        InvalidArgument,
        "Output range overflows size_t");
    ET_CHECK_OK_OR_RETURN_ERROR(
        write_all(fd_, data.data(), data.size(), offset, temporary_path_));
    extent_ = std::max(extent_, end);
    return Error::Ok;
  }

  Error finish() {
    if (finished_) {
      return Error::Ok;
    }
    ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Writer is not active");
    ET_CHECK_OK_OR_RETURN_ERROR(resize_file(fd_, extent_, temporary_path_));
    ET_CHECK_OK_OR_RETURN_ERROR(sync_file(fd_, temporary_path_));
    if (::close(fd_) != 0) {
      fd_ = -1;
      return log_io_error("close", temporary_path_);
    }
    fd_ = -1;
    finished_ = true;
    return Error::Ok;
  }

  const std::string& temporary_path() const {
    return temporary_path_;
  }

  void mark_published() {
    temporary_path_.clear();
  }

 private:
  void discard() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (!temporary_path_.empty()) {
#if defined(_WIN64)
      ::_unlink(temporary_path_.c_str());
#else
      ::unlink(temporary_path_.c_str());
#endif
      temporary_path_.clear();
    }
  }

  int fd_;
  std::string temporary_path_;
  size_t extent_{0};
  bool finished_{false};
};

static std::unique_ptr<FileDataWriterState> create_temporary_file(
    const std::string& destination_path,
    const std::string& requested_directory) {
  const std::string directory = requested_directory.empty()
      ? parent_directory(destination_path)
      : requested_directory;
  const std::string prefix = "." + base_name(destination_path) + ".et-";

#if defined(_WIN64)
  char temporary_path[MAX_PATH];
  if (::GetTempFileNameA(directory.c_str(), "etw", 0, temporary_path) == 0) {
    errno = EACCES;
    log_io_error("create temporary file", directory);
    return nullptr;
  }
  const int fd = ::_open(
      temporary_path, _O_BINARY | _O_RDWR | _O_TRUNC, _S_IREAD | _S_IWRITE);
  if (fd < 0) {
    ::DeleteFileA(temporary_path);
    log_io_error("open temporary file", temporary_path);
    return nullptr;
  }
  return std::make_unique<FileDataWriterState>(fd, temporary_path);
#else
  std::string path = directory + "/" + prefix + "XXXXXX";
  std::vector<char> mutable_path(path.begin(), path.end());
  mutable_path.push_back('\0');
  const int fd = ::mkstemp(mutable_path.data());
  if (fd < 0) {
    log_io_error("create temporary file", directory);
    return nullptr;
  }

  struct stat source_stat;
  if (::stat(destination_path.c_str(), &source_stat) == 0 &&
      ::fchmod(fd, source_stat.st_mode & 07777) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    ::unlink(mutable_path.data());
    errno = saved_errno;
    log_io_error("copy file permissions", destination_path);
    return nullptr;
  }
  return std::make_unique<FileDataWriterState>(fd, mutable_path.data());
#endif
}

FileDataWriter::FileDataWriter(
    std::string file_name,
    std::string temporary_directory)
    : file_name_(std::move(file_name)),
      temporary_directory_(std::move(temporary_directory)) {}

FileDataWriter::~FileDataWriter() = default;

Error FileDataWriter::write(Span<const uint8_t> data, size_t offset) {
  ET_CHECK_OR_RETURN_ERROR(
      !published_, InvalidState, "Cannot write after publication");
  ET_CHECK_OR_RETURN_ERROR(
      !file_name_.empty(), InvalidArgument, "File name cannot be empty");
  if (state_ == nullptr) {
    state_ = create_temporary_file(file_name_, temporary_directory_);
    if (state_ == nullptr) {
      return Error::AccessFailed;
    }
  }
  return state_->write(data, offset);
}

Error FileDataWriter::publish() {
  ET_CHECK_OR_RETURN_ERROR(
      !published_, InvalidState, "Writer has already published");
  ET_CHECK_OR_RETURN_ERROR(
      !file_name_.empty(), InvalidArgument, "File name cannot be empty");
  if (state_ == nullptr) {
    state_ = create_temporary_file(file_name_, temporary_directory_);
    if (state_ == nullptr) {
      return Error::AccessFailed;
    }
  }
  ET_CHECK_OK_OR_RETURN_ERROR(state_->finish());

  bool replaced = false;
  const Error error =
      publish_file(state_->temporary_path(), file_name_, &replaced);
  if (replaced) {
    published_ = true;
    state_->mark_published();
  }
  return error;
}

} // namespace extension
} // namespace executorch
