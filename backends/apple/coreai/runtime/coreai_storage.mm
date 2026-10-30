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
#include <utility>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

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
        is_storage_component(component),
        InvalidArgument,
        "Invalid storage path");
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
    FileDescriptor next = std::move(*opened->descriptor);
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(parent.get()));
    parent = std::move(next);
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
      const ssize_t count = fault_aware_retry(FaultPoint::Write, [&] {
        return write(file.get(), bytes, std::min(size, size_t(1024 * 1024)));
      });
      ET_CHECK_OR_RETURN_ERROR(
          count > 0, AccessFailed, "Cannot write Core AI storage file");
      bytes += count;
      size -= count;
    }
    ET_CHECK_OR_RETURN_ERROR(
        fault_aware_retry(
            FaultPoint::SyncFile, [&] { return fsync(file.get()); }) == 0 &&
            (sync_mode != StorageSyncMode::Full ||
             fault_aware_retry(
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

enum class StorageTreePass {
  Preflight,
  Remove,
};

enum class StorageEntryState {
  Missing,
  NonDirectory,
  Directory,
};

struct InspectedStorageEntry final {
  StorageEntryState state;
  std::optional<FileDescriptor> directory;
  struct stat info;
};

Result<InspectedStorageEntry> inspect_storage_entry(
    int parent_fd,
    NSString* name,
    StorageRemovalPolicy policy) {
  ET_CHECK_OR_RETURN_ERROR(
      is_storage_component(name),
      InvalidArgument,
      "Invalid Core AI entry name");
  struct stat info {};
  const int inspected =
      lstat_at(parent_fd, name.fileSystemRepresentation, &info);
  if (inspected < 0 && errno == ENOENT) {
    return InspectedStorageEntry{
        StorageEntryState::Missing, std::nullopt, info};
  }
  ET_CHECK_OR_RETURN_ERROR(
      inspected == 0, AccessFailed, "Cannot inspect Core AI storage entry");
  if (!S_ISDIR(info.st_mode)) {
    ET_CHECK_OR_RETURN_ERROR(
        policy == StorageRemovalPolicy::Maintenance,
        InvalidExternalData,
        "Core AI storage entry is not a directory");
    return InspectedStorageEntry{
        StorageEntryState::NonDirectory, std::nullopt, info};
  }
  FileDescriptor entry(
      open_directory_at(parent_fd, name.fileSystemRepresentation));
  if (entry.get() < 0 && errno == ENOENT) {
    return InspectedStorageEntry{
        StorageEntryState::Missing, std::nullopt, info};
  }
  ET_CHECK_OR_RETURN_ERROR(
      entry.get() >= 0, AccessFailed, "Cannot open Core AI storage directory");
  return InspectedStorageEntry{
      StorageEntryState::Directory,
      std::optional<FileDescriptor>(std::move(entry)),
      info};
}

bool same_storage_entry(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
      (left.st_mode & S_IFMT) == (right.st_mode & S_IFMT);
}

Error walk_storage_tree(
    int fd,
    StorageTreePass pass,
    StorageRemovalPolicy policy) {
  auto children = storage_children(fd);
  if (!children.ok()) {
    return children.error();
  }
  for (NSString* name in children.get()) {
    struct stat info;
    const int inspected = lstat_at(fd, name.fileSystemRepresentation, &info);
    if (inspected < 0 && errno == ENOENT) {
      continue;
    }
    ET_CHECK_OR_RETURN_ERROR(
        inspected == 0, AccessFailed, "Cannot inspect Core AI storage entry");
    const bool directory = S_ISDIR(info.st_mode);
    if (policy == StorageRemovalPolicy::Strict) {
      ET_CHECK_OR_RETURN_ERROR(
          directory || S_ISREG(info.st_mode),
          InvalidExternalData,
          "Unexpected Core AI storage entry type");
    }
    if (directory) {
      FileDescriptor child(
          open_directory_at(fd, name.fileSystemRepresentation));
      if (child.get() < 0 && errno == ENOENT) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          child.get() >= 0,
          AccessFailed,
          "Cannot open Core AI storage directory");
      ET_CHECK_OK_OR_RETURN_ERROR(walk_storage_tree(child.get(), pass, policy));
    }
    if (pass == StorageTreePass::Remove) {
      struct stat current;
      const int revalidated =
          lstat_at(fd, name.fileSystemRepresentation, &current);
      if (revalidated < 0 && errno == ENOENT) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          revalidated == 0 && same_storage_entry(info, current),
          AccessFailed,
          "Core AI storage entry changed during removal");
      const int removed = fault_aware_retry(FaultPoint::Remove, [&] {
        return unlinkat(
            fd, name.fileSystemRepresentation, directory ? AT_REMOVEDIR : 0);
      });
      if (removed < 0 && errno == ENOENT) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          removed == 0, AccessFailed, "Cannot remove Core AI storage entry");
    }
  }
  return pass == StorageTreePass::Remove ? sync_storage_directory(fd)
                                         : Error::Ok;
}
} // namespace

Result<StorageDirectory> open_storage_directory(
    int parent_fd,
    NSString* name,
    StorageCreationMode mode) {
  ET_CHECK_OR_RETURN_ERROR(
      is_storage_component(name),
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
    return StorageDirectory{std::nullopt, false};
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0, AccessFailed, "Cannot open Core AI storage directory");
  return StorageDirectory{FileDescriptor(fd), created};
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

Error preflight_storage_directory(
    int parent_fd,
    NSString* name,
    StorageRemovalPolicy policy) {
  auto entry = inspect_storage_entry(parent_fd, name, policy);
  if (!entry.ok()) {
    return entry.error();
  }
  return entry->state == StorageEntryState::Directory
      ? walk_storage_tree(
            entry->directory->get(), StorageTreePass::Preflight, policy)
      : Error::Ok;
}

Error remove_storage_directory(
    int parent_fd,
    NSString* name,
    StorageRemovalPolicy policy) {
  auto entry = inspect_storage_entry(parent_fd, name, policy);
  if (!entry.ok()) {
    return entry.error();
  }
  if (entry->state == StorageEntryState::Missing) {
    return Error::Ok;
  }
  if (entry->state == StorageEntryState::NonDirectory) {
    struct stat current;
    const int revalidated =
        lstat_at(parent_fd, name.fileSystemRepresentation, &current);
    if (revalidated < 0 && errno == ENOENT) {
      return Error::Ok;
    }
    ET_CHECK_OR_RETURN_ERROR(
        revalidated == 0 && same_storage_entry(entry->info, current),
        AccessFailed,
        "Core AI storage entry changed during removal");
    const int removed = fault_aware_retry(FaultPoint::Remove, [&] {
      return unlinkat(parent_fd, name.fileSystemRepresentation, 0);
    });
    ET_CHECK_OR_RETURN_ERROR(
        removed == 0 || (removed < 0 && errno == ENOENT),
        AccessFailed,
        "Cannot remove Core AI storage entry");
    return sync_storage_directory(parent_fd);
  }
  ET_CHECK_OK_OR_RETURN_ERROR(walk_storage_tree(
      entry->directory->get(), StorageTreePass::Preflight, policy));
  ET_CHECK_OK_OR_RETURN_ERROR(walk_storage_tree(
      entry->directory->get(), StorageTreePass::Remove, policy));
  struct stat current;
  const int revalidated =
      lstat_at(parent_fd, name.fileSystemRepresentation, &current);
  if (revalidated < 0 && errno == ENOENT) {
    return sync_storage_directory(parent_fd);
  }
  ET_CHECK_OR_RETURN_ERROR(
      revalidated == 0 && same_storage_entry(entry->info, current),
      AccessFailed,
      "Core AI storage directory changed during removal");
  const int removed = fault_aware_retry(FaultPoint::Remove, [&] {
    return unlinkat(parent_fd, name.fileSystemRepresentation, AT_REMOVEDIR);
  });
  ET_CHECK_OR_RETURN_ERROR(
      removed == 0 || (removed < 0 && errno == ENOENT),
      AccessFailed,
      "Cannot remove Core AI storage directory");
  return sync_storage_directory(parent_fd);
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
      fault_aware_retry(FaultPoint::SyncDirectory, [&] { return fsync(fd); }) ==
          0,
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
      is_storage_component(name) && data != nil,
      InvalidArgument,
      "Invalid Core AI storage publication");
  NSString* temporary =
      [@".tmp-" stringByAppendingString:NSUUID.UUID.UUIDString];
  // Bookmarks are small and costly to lose, so they get a full device flush.
  auto result = write_file(
      root_fd, temporary, data.bytes, data.length, StorageSyncMode::Full);
  if (result == Error::Ok && fault_aware_retry(FaultPoint::Rename, [&] {
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
        fault_aware_retry(
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
