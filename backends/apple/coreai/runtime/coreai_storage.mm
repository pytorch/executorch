/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_storage.h"
#include "coreai_fault_injection.h"
#include "coreai_file.h"
#include "coreai_validation.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

template <typename F>
auto storage_call(FaultPoint point, F call) {
  return retry_eintr([&]() -> decltype(call()) {
    const int fault = fault_point(point);
    if (fault != 0) {
      errno = fault;
      return -1;
    }
    return call();
  });
}

enum class StorageSyncMode {
  Normal,
  Full,
};

Error write_file(
    int root_fd,
    NSString* relative_path,
    const void* data,
    size_t size,
    StorageSyncMode sync_mode) {
  NSArray<NSString*>* components =
      [relative_path componentsSeparatedByString:@"/"];
  ET_CHECK_OR_RETURN_ERROR(
      components.count > 0 && (data != nullptr || size == 0),
      InvalidArgument,
      "Invalid Core AI storage write");
  for (NSString* component in components) {
    ET_CHECK_OR_RETURN_ERROR(
        storage_component(component), InvalidArgument, "Invalid storage path");
  }
  FileDescriptor parent(
      retry_eintr([&] { return fcntl(root_fd, F_DUPFD_CLOEXEC, 0); }));
  ET_CHECK_OR_RETURN_ERROR(
      parent.get() >= 0, AccessFailed, "Cannot open storage parent");
  for (NSUInteger i = 0; i + 1 < components.count; ++i) {
    auto opened = open_storage_directory(
        parent.get(), components[i], StorageCreationMode::Create);
    if (!opened.ok())
      return opened.error();
    FileDescriptor next(opened.get().fd);
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(parent.get()));
    parent.reset(next.release());
  }
  FileDescriptor file(retry_eintr([&] {
    return openat(
        parent.get(),
        components.lastObject.fileSystemRepresentation,
        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
        0600);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      file.get() >= 0, AccessFailed, "Cannot create Core AI storage file");
  const auto result = [&]() -> Error {
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size > 0) {
      const ssize_t count = storage_call(FaultPoint::Write, [&] {
        return write(file.get(), bytes, std::min(size, size_t(1024 * 1024)));
      });
      ET_CHECK_OR_RETURN_ERROR(
          count > 0, AccessFailed, "Cannot write Core AI storage file");
      bytes += count;
      size -= count;
    }
    ET_CHECK_OR_RETURN_ERROR(
        storage_call(FaultPoint::SyncFile, [&] { return fsync(file.get()); }) ==
                0 &&
            (sync_mode != StorageSyncMode::Full ||
             storage_call(
                 FaultPoint::FullSync,
                 [&] { return fcntl(file.get(), F_FULLFSYNC); }) == 0),
        AccessFailed,
        "Cannot synchronize Core AI file");
    ET_CHECK_OR_RETURN_ERROR(
        ::close(file.release()) == 0,
        AccessFailed,
        "Cannot close Core AI storage file");
    return sync_storage_directory(parent.get());
  }();
  if (result != Error::Ok) {
    if (retry_eintr([&] {
          return unlinkat(
              parent.get(), components.lastObject.fileSystemRepresentation, 0);
        }) == 0) {
      (void)sync_storage_directory(parent.get());
    } else {
      ET_LOG(
          Error, "Cannot remove failed Core AI storage write: errno=%d", errno);
    }
  }
  return result;
}
} // namespace

Result<StorageDirectory> open_storage_directory(
    int parent_fd,
    NSString* name,
    StorageCreationMode mode) {
  ET_CHECK_OR_RETURN_ERROR(
      storage_component(name),
      InvalidArgument,
      "Invalid storage directory name");
  const auto open_directory = [&] {
    return open_directory_at(parent_fd, name.fileSystemRepresentation);
  };
  const bool create = mode == StorageCreationMode::Create;
  int fd = open_directory();
  bool created = false;
  if (fd < 0 && errno == ENOENT && create) {
    ET_CHECK_OR_RETURN_ERROR(
        fault_point(FaultPoint::BeforeCreateDirectory) == 0,
        AccessFailed,
        "Cannot begin Core AI directory creation");
    const int made = retry_eintr([&] {
      return mkdirat(parent_fd, name.fileSystemRepresentation, 0777);
    });
    ET_CHECK_OR_RETURN_ERROR(
        made == 0 || errno == EEXIST,
        AccessFailed,
        "Cannot create Core AI storage directory");
    created = made == 0;
    fd = open_directory();
  }
  if (fd < 0 && errno == ENOENT && !create)
    return StorageDirectory{-1, false};
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0, AccessFailed, "Cannot open Core AI storage directory");
  return StorageDirectory{fd, created};
}

Result<NSArray<NSString*>*> storage_children(int fd, StorageNamePolicy policy) {
  // Reopening avoids sharing the caller's directory offset.
  FileDescriptor copy(retry_eintr(
      [&] { return openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); }));
  ET_CHECK_OR_RETURN_ERROR(
      copy.get() >= 0, AccessFailed, "Cannot open storage directory");
  DIR* directory = fdopendir(copy.get());
  ET_CHECK_OR_RETURN_ERROR(
      directory != nullptr,
      AccessFailed,
      "Cannot open storage directory stream");
  copy.release();
  NSMutableArray<NSString*>* names = [NSMutableArray array];
  Error result = Error::Ok;
  while (true) {
    errno = 0;
    dirent* entry = readdir(directory);
    if (entry == nullptr) {
      if (errno == EINTR)
        continue;
      if (errno != 0)
        result = Error::AccessFailed;
      break;
    }
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    NSString* name = [[NSString alloc] initWithBytes:entry->d_name
                                              length:strlen(entry->d_name)
                                            encoding:NSUTF8StringEncoding];
    if (name == nil) {
      if (policy == StorageNamePolicy::SkipInvalid) {
        continue;
      }
      result = Error::InvalidExternalData;
      break;
    }
    [names addObject:name];
  }
  const int closed = closedir(directory);
  ET_CHECK_OK_OR_RETURN_ERROR(result);
  ET_CHECK_OR_RETURN_ERROR(
      closed == 0, AccessFailed, "Cannot close storage directory stream");
  return [names sortedArrayUsingSelector:@selector(compare:)];
}

int open_storage_shared_file(int parent_fd, const char* name) {
  // APFS can return ENOENT to a concurrent O_CREAT loser without O_EXCL.
  int fd = retry_eintr([&] {
    return openat(
        parent_fd,
        name,
        O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
        0600);
  });
  if (fd < 0 && errno == EEXIST) {
    fd = retry_eintr([&] {
      return openat(parent_fd, name, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    });
  }
  return fd;
}

Error sync_storage_directory(int fd) {
  ET_CHECK_OR_RETURN_ERROR(
      storage_call(FaultPoint::SyncDirectory, [&] { return fsync(fd); }) == 0,
      AccessFailed,
      "Cannot synchronize Core AI directory");
  return Error::Ok;
}

Error write_storage_file(
    int root_fd,
    NSString* relative_path,
    const void* data,
    size_t size) {
  return write_file(
      root_fd, relative_path, data, size, StorageSyncMode::Normal);
}

Error publish_storage_data(int root_fd, NSString* name, NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      storage_component(name) && data != nil,
      InvalidArgument,
      "Invalid Core AI storage publication");
  NSString* temporary =
      [@".tmp-" stringByAppendingString:NSUUID.UUID.UUIDString];
  // Bookmarks are small and costly to lose, so they get a full device flush.
  auto result = write_file(
      root_fd, temporary, data.bytes, data.length, StorageSyncMode::Full);
  if (result == Error::Ok && storage_call(FaultPoint::Rename, [&] {
                               return renameat(
                                   root_fd,
                                   temporary.fileSystemRepresentation,
                                   root_fd,
                                   name.fileSystemRepresentation);
                             }) != 0) {
    ET_LOG(Error, "Cannot publish Core AI storage data: errno=%d", errno);
    result = Error::AccessFailed;
  }
  if (result == Error::Ok) {
    // Flush the renamed entry as well as the file's pre-rename contents.
    ET_CHECK_OR_RETURN_ERROR(
        storage_call(
            FaultPoint::FullSync,
            [&] { return fcntl(root_fd, F_FULLFSYNC); }) == 0,
        AccessFailed,
        "Cannot fully synchronize Core AI publication directory");
    return Error::Ok;
  }
  if (unlinkat(root_fd, temporary.fileSystemRepresentation, 0) == 0) {
    (void)sync_storage_directory(root_fd);
  }
  return result;
}

} // namespace executorch::backends::coreai
