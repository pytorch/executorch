/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_source_bundle.h"

#import "coreai_cache_entry.h"
#import "coreai_fault_injection.h"
#import "coreai_manifest.h"
#import "coreai_storage.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <utility>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

enum class EntryState {
  Missing,
  Valid,
  Corrupt,
};

Error path_error(int error) {
  return error == ENOENT || error == ENOTDIR || error == ELOOP
      ? Error::InvalidExternalData
      : Error::AccessFailed;
}

Error validate_directory(
    int fd,
    NSString* relative,
    NSDictionary<NSString*, NSNumber*>* files,
    NSSet<NSString*>* expected_directories,
    NSMutableSet<NSString*>* actual_files) {
  auto children = storage_children(fd);
  if (!children.ok()) {
    return children.error();
  }
  for (NSString* name in children.get()) {
    @autoreleasepool {
      NSString* path = [relative stringByAppendingPathComponent:name]
                           .decomposedStringWithCanonicalMapping;
      struct stat info;
      if (lstat_at(fd, name.fileSystemRepresentation, &info) != 0) {
        return path_error(errno);
      }
      const bool directory = S_ISDIR(info.st_mode);
      ET_CHECK_OR_RETURN_ERROR(
          (directory && [expected_directories containsObject:path]) ||
              (S_ISREG(info.st_mode) && files[path] != nil),
          InvalidExternalData,
          "Unexpected Core AI bundle entry: %s",
          path.UTF8String);
      if (directory) {
        FileDescriptor child(
            open_directory_at(fd, name.fileSystemRepresentation));
        if (child.get() < 0) {
          return path_error(errno);
        }
        ET_CHECK_OK_OR_RETURN_ERROR(validate_directory(
            child.get(), path, files, expected_directories, actual_files));
      } else {
        ET_CHECK_OR_RETURN_ERROR(
            static_cast<uint64_t>(info.st_size) ==
                files[path].unsignedLongLongValue,
            InvalidExternalData,
            "Core AI asset size does not match manifest");
        [actual_files addObject:path];
      }
    }
  }
  return Error::Ok;
}

// Filesystem conversion decomposes Unicode without changing NamedDataMap keys.
NSDictionary<NSString*, NSNumber*>* normalized_files(
    NSDictionary<NSString*, NSNumber*>* files) {
  NSMutableDictionary<NSString*, NSNumber*>* normalized =
      [NSMutableDictionary dictionaryWithCapacity:files.count];
  for (NSString* file in files) {
    normalized[file.decomposedStringWithCanonicalMapping] = files[file];
  }
  return normalized;
}

Error validate_tree(
    int fd,
    NSDictionary<NSString*, NSNumber*>* selected_files) {
  NSDictionary<NSString*, NSNumber*>* files = normalized_files(selected_files);
  NSMutableSet<NSString*>* expected_directories = [NSMutableSet set];
  for (NSString* file in files) {
    NSString* directory = file.stringByDeletingLastPathComponent;
    while (directory.length > 0) {
      [expected_directories addObject:directory];
      directory = directory.stringByDeletingLastPathComponent;
    }
  }
  NSMutableSet<NSString*>* actual_files = [NSMutableSet set];
  ET_CHECK_OK_OR_RETURN_ERROR(
      validate_directory(fd, @"", files, expected_directories, actual_files));
  ET_CHECK_OR_RETURN_ERROR(
      [actual_files isEqualToSet:[NSSet setWithArray:files.allKeys]],
      InvalidExternalData,
      "Core AI bundle file set does not match manifest");
  return Error::Ok;
}

Result<EntryState> inspect_source_bundle(
    int sources_fd,
    NSString* key,
    NSDictionary<NSString*, NSNumber*>* files) {
  FileDescriptor entry(
      open_directory_at(sources_fd, key.fileSystemRepresentation));
  if (entry.get() < 0) {
    if (errno == ENOENT) {
      return EntryState::Missing;
    }
    ET_CHECK_OR_RETURN_ERROR(
        errno == ENOTDIR || errno == ELOOP,
        AccessFailed,
        "Cannot inspect Core AI stored bundle");
    return EntryState::Corrupt;
  }
  const Error error = validate_tree(entry.get(), files);
  if (error == Error::Ok) {
    return EntryState::Valid;
  }
  if (error == Error::InvalidExternalData) {
    return EntryState::Corrupt;
  }
  return error;
}

// Removes the unpublished build directory on every early return.
struct PendingDirectory {
  int parent_fd;
  NSString* name;
  bool published = false;

  PendingDirectory(int parent_fd, NSString* name)
      : parent_fd(parent_fd), name(name) {}
  PendingDirectory(const PendingDirectory&) = delete;
  PendingDirectory& operator=(const PendingDirectory&) = delete;
  PendingDirectory(PendingDirectory&& other)
      : parent_fd(other.parent_fd),
        name(other.name),
        published(other.published) {
    other.published = true;
  }

  ~PendingDirectory() {
    if (published) {
      return;
    }
    const Error error =
        remove_storage_directory(parent_fd, name, StorageRemovalPolicy::Strict);
    if (error != Error::Ok) {
      ET_LOG(Error, "Cannot remove pending Core AI source directory");
    }
  }
};

Result<PendingDirectory> create_pending_directory(int sources_fd) {
  NSString* name =
      [@".pending-" stringByAppendingString:NSUUID.UUID.UUIDString];
  ET_CHECK_OR_RETURN_ERROR(
      fault_aware_retry(
          FaultPoint::CreatePendingDirectory,
          [&] {
            return mkdirat(sources_fd, name.fileSystemRepresentation, 0777);
          }) == 0,
      AccessFailed,
      "Cannot create pending Core AI source directory");
  return PendingDirectory{sources_fd, [name copy]};
}

Result<FileDescriptor> open_pending_directory(int sources_fd, NSString* name) {
  FileDescriptor fd(fault_aware_retry(FaultPoint::OpenPendingDirectory, [&] {
    return open_directory_at(sources_fd, name.fileSystemRepresentation);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      fd.get() >= 0,
      AccessFailed,
      "Cannot open pending Core AI source directory");
  return fd;
}

Error write_pending_bundle(
    int pending_fd,
    const SelectedBundle& bundle,
    const runtime::NamedDataMap& named_data) {
  for (NSString* file in bundle.files) {
    @autoreleasepool {
      NSString* data_key =
          [NSString stringWithFormat:@"coreai/%@/%@", bundle.export_hash, file];
      auto buffer = named_data.get_data(data_key.UTF8String);
      if (!buffer.ok()) {
        ET_CHECK_OR_RETURN_ERROR(
            buffer.error() != Error::NotFound,
            InvalidExternalData,
            "Missing selected Core AI named data: %s",
            data_key.UTF8String);
        return buffer.error();
      }
      ET_CHECK_OR_RETURN_ERROR(
          buffer->size() == bundle.files[file].unsignedLongLongValue &&
              (buffer->size() == 0 || buffer->data() != nullptr),
          InvalidExternalData,
          "Invalid Core AI named data buffer");
      ET_CHECK_OK_OR_RETURN_ERROR(
          write_storage_file(pending_fd, file, buffer->data(), buffer->size()));
    }
  }
  ET_CHECK_OK_OR_RETURN_ERROR(validate_tree(pending_fd, bundle.files));
  return full_sync_storage_directory(pending_fd);
}

Result<bool> publish_pending_directory(
    int sources_fd,
    PendingDirectory& pending,
    NSString* key) {
  const int result = fault_aware_retry(FaultPoint::Rename, [&] {
    return renameatx_np(
        sources_fd,
        pending.name.fileSystemRepresentation,
        sources_fd,
        key.fileSystemRepresentation,
        RENAME_EXCL);
  });
  if (result == 0) {
    pending.published = true;
    ET_CHECK_OK_OR_RETURN_ERROR(full_sync_storage_directory(sources_fd));
    return true;
  }
  ET_CHECK_OR_RETURN_ERROR(
      errno == EEXIST, AccessFailed, "Cannot publish Core AI stored bundle");
  return false;
}

Error recover_lost_publication_race(
    int sources_fd,
    NSString* key,
    NSDictionary<NSString*, NSNumber*>* files) {
  auto winner = inspect_source_bundle(sources_fd, key, files);
  if (!winner.ok()) {
    return winner.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      winner.get() == EntryState::Valid,
      InvalidExternalData,
      "Concurrently published Core AI source is incomplete");
  return full_sync_storage_directory(sources_fd);
}

} // namespace

Result<NSURL*> prepare_source_bundle(
    const SelectedBundle& bundle,
    const runtime::NamedDataMap* named_data,
    const LockedCacheEntry& cache_entry,
    CorruptBundlePolicy corrupt_policy) {
  NSString* key = cache_entry.key();
  auto sources = cache_entry.prepare_source_directory();
  if (!sources.ok()) {
    return sources.error();
  }
  NSURL* model_url = [[sources->url URLByAppendingPathComponent:key
                                                    isDirectory:YES]
      URLByAppendingPathComponent:bundle.bundle_name
                      isDirectory:YES];
  FileDescriptor sources_fd = std::move(sources->descriptor);

  auto state = inspect_source_bundle(sources_fd.get(), key, bundle.files);
  if (!state.ok()) {
    return state.error();
  }
  if (state.get() == EntryState::Valid) {
    ET_CHECK_OK_OR_RETURN_ERROR(full_sync_storage_directory(sources_fd.get()));
    return model_url;
  }
  if (state.get() == EntryState::Corrupt) {
    ET_CHECK_OR_RETURN_ERROR(
        corrupt_policy == CorruptBundlePolicy::Replace && named_data != nullptr,
        InvalidExternalData,
        "Cannot replace a damaged Core AI stored bundle");
    ET_CHECK_OK_OR_RETURN_ERROR(remove_storage_directory(
        sources_fd.get(), key, StorageRemovalPolicy::Maintenance));
  }
  ET_CHECK_OR_RETURN_ERROR(
      named_data != nullptr,
      InvalidProgram,
      "Missing NamedDataMap for Core AI source recovery");

  auto created = create_pending_directory(sources_fd.get());
  if (!created.ok()) {
    return created.error();
  }
  PendingDirectory pending = std::move(created.get());
  auto pending_fd = open_pending_directory(sources_fd.get(), pending.name);
  if (!pending_fd.ok()) {
    return pending_fd.error();
  }
  ET_CHECK_OK_OR_RETURN_ERROR(
      write_pending_bundle(pending_fd->get(), bundle, *named_data));

  auto published = publish_pending_directory(sources_fd.get(), pending, key);
  if (!published.ok()) {
    return published.error();
  }
  if (!published.get()) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        recover_lost_publication_race(sources_fd.get(), key, bundle.files));
  }
  return model_url;
}

} // namespace executorch::backends::coreai
