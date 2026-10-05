/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_assets.h"
#include "coreai_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#import "coreai_storage.h"

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Error path_error(int error) {
  return error == ENOENT || error == ENOTDIR || error == ELOOP
      ? Error::InvalidExternalData
      : Error::AccessFailed;
}

bool nonempty_string(id value) {
  return [value isKindOfClass:NSString.class] && [value length] > 0 &&
      [value rangeOfString:@"\0"].location == NSNotFound;
}

bool names_array(id value) {
  if (![value isKindOfClass:NSArray.class]) {
    return false;
  }
  NSMutableSet<NSString*>* seen = [NSMutableSet set];
  for (id name in value) {
    if (!nonempty_string(name) || [seen containsObject:name]) {
      return false;
    }
    [seen addObject:name];
  }
  return true;
}

bool relative_path(NSString* path) {
  if (!nonempty_string(path) || path.isAbsolutePath ||
      [path containsString:@"\\"]) {
    return false;
  }
  for (NSString* component in [path componentsSeparatedByString:@"/"]) {
    if (component.length == 0 || [component isEqualToString:@"."] ||
        [component isEqualToString:@".."]) {
      return false;
    }
  }
  return true;
}

bool file_size(id value) {
  return [value isKindOfClass:NSNumber.class] &&
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
      !CFNumberIsFloatType((__bridge CFNumberRef)value) &&
      [value compare:@0] != NSOrderedAscending &&
      [value compare:@(std::numeric_limits<int64_t>::max())] !=
      NSOrderedDescending &&
      [value isEqualToNumber:@([value unsignedLongLongValue])];
}

Error validate_files(
    NSDictionary<NSString*, NSNumber*>* filenames,
    NSSet<NSString*>* bundles) {
  ET_CHECK_OR_RETURN_ERROR(
      [filenames isKindOfClass:NSDictionary.class] && filenames.count > 0,
      InvalidProgram,
      "Core AI manifest requires asset file sizes");
  NSSet<NSString*>* files = [NSSet setWithArray:filenames.allKeys];
  NSMutableSet<NSString*>* populated = [NSMutableSet set];
  for (NSString* file in filenames) {
    ET_CHECK_OR_RETURN_ERROR(
        relative_path(file) && file_size(filenames[file]) &&
            [bundles containsObject:[file componentsSeparatedByString:@"/"]
                                        .firstObject] &&
            [file containsString:@"/"] &&
            [file canBeConvertedToEncoding:NSUTF8StringEncoding],
        InvalidProgram,
        "Invalid Core AI asset path or size");
    [populated addObject:[file componentsSeparatedByString:@"/"].firstObject];
    NSString* parent = file.stringByDeletingLastPathComponent;
    while (parent.length > 0) {
      ET_CHECK_OR_RETURN_ERROR(
          ![files containsObject:parent],
          InvalidProgram,
          "Conflicting Core AI asset paths");
      parent = parent.stringByDeletingLastPathComponent;
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      [populated isEqualToSet:bundles],
      InvalidProgram,
      "Core AI manifest requires files for every declared bundle");
  return Error::Ok;
}

bool parse_version(id value, NSOperatingSystemVersion& version) {
  // A null floor means the exporter used the SDK's save_asset default.
  if (value == NSNull.null) {
    version = {};
    return true;
  }
  if (!nonempty_string(value)) {
    return false;
  }
  NSArray<NSString*>* parts = [value componentsSeparatedByString:@"."];
  if (parts.count > 3) {
    return false;
  }
  NSInteger numbers[3] = {0, 0, 0};
  NSCharacterSet* invalid = [[NSCharacterSet
      characterSetWithCharactersInString:@"0123456789"] invertedSet];
  for (NSUInteger i = 0; i < parts.count; ++i) {
    NSString* part = parts[i];
    if (part.length == 0 || part.length > 6 ||
        [part rangeOfCharacterFromSet:invalid].location != NSNotFound) {
      return false;
    }
    numbers[i] = part.integerValue;
  }
  version = {numbers[0], numbers[1], numbers[2]};
  return numbers[0] > 0;
}

Error validate_directory(int fd, NSString* relative,
                         NSDictionary<NSString*, NSNumber*>* files,
                         NSSet<NSString*>* expected_directories,
                         NSMutableSet<NSString*>* actual_files) {
  auto children = storage_children(fd);
  if (!children.ok()) {
    return children.error();
  }
  for (NSString* name in children.get()) {
    @autoreleasepool {
      NSString* path = [relative stringByAppendingPathComponent:name];
      struct stat info;
      if (retry_eintr([&] {
            return fstatat(fd, name.fileSystemRepresentation, &info,
                           AT_SYMLINK_NOFOLLOW);
          }) != 0) {
        return path_error(errno);
      }
      const bool directory = S_ISDIR(info.st_mode);
      ET_CHECK_OR_RETURN_ERROR(
          (directory && [expected_directories containsObject:path]) ||
              (S_ISREG(info.st_mode) && files[path] != nil),
          InvalidExternalData, "Unexpected Core AI bundle entry: %s",
          path.UTF8String);
      if (directory) {
        FileDescriptor child(retry_eintr([&] {
          return openat(fd, name.fileSystemRepresentation,
                        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }));
        if (child.get() < 0) {
          return path_error(errno);
        }
        ET_CHECK_OK_OR_RETURN_ERROR(validate_directory(
            child.get(), path, files, expected_directories, actual_files));
      } else {
        ET_CHECK_OR_RETURN_ERROR(
            static_cast<uint64_t>(info.st_size) ==
                files[path].unsignedLongLongValue,
            InvalidExternalData, "Core AI asset size does not match manifest");
        [actual_files addObject:path];
      }
    }
  }
  return Error::Ok;
}

// Checks the exact file set, entry types and sizes. Contents are not hashed.
Error validate_tree(int fd, NSDictionary<NSString*, NSNumber*>* files) {
  NSMutableSet<NSString*>* expected_directories = [NSMutableSet set];
  for (NSString* file in files) {
    NSString* directory = file.stringByDeletingLastPathComponent;
    while (directory.length > 0) {
      [expected_directories addObject:directory];
      directory = directory.stringByDeletingLastPathComponent;
    }
  }
  NSMutableSet<NSString*>* actual_files = [NSMutableSet set];
  ET_CHECK_OK_OR_RETURN_ERROR(validate_directory(
      fd, @"", files, expected_directories, actual_files));
  ET_CHECK_OR_RETURN_ERROR(
      [actual_files isEqualToSet:[NSSet setWithArray:files.allKeys]],
      InvalidExternalData, "Core AI bundle file set does not match manifest");
  return Error::Ok;
}

enum class EntryState { Missing, Valid, Corrupt };

Result<EntryState> inspect_entry(
    int root_fd,
    NSString* name,
    NSDictionary<NSString*, NSNumber*>* files) {
  FileDescriptor entry(retry_eintr([&] {
    return openat(root_fd, name.fileSystemRepresentation,
                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  }));
  if (entry.get() < 0) {
    if (errno == ENOENT) return EntryState::Missing;
    ET_CHECK_OR_RETURN_ERROR(errno == ENOTDIR || errno == ELOOP, AccessFailed,
                             "Cannot inspect Core AI stored bundle");
    return EntryState::Corrupt;
  }
  const auto error = validate_tree(entry.get(), files);
  if (error == Error::Ok) {
    return EntryState::Valid;
  }
  if (error == Error::InvalidExternalData) {
    return EntryState::Corrupt;
  }
  return error;
}

struct StagingDirectory {
  NSURL* url;
  ~StagingDirectory() {
    if (url != nil) {
      NSError* error = nil;
      if (![NSFileManager.defaultManager removeItemAtURL:url error:&error]) {
        ET_LOG(
            Error,
            "Cannot remove Core AI staging directory: %s",
            error.localizedDescription.UTF8String);
      }
    }
  }
};
} // namespace

Result<Manifest> parse_manifest(NSData* data) {
  NSError* error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data
                                              options:0
                                                error:&error];
  ET_CHECK_OR_RETURN_ERROR(
      [object isKindOfClass:NSDictionary.class],
      InvalidProgram,
      "Invalid Core AI manifest JSON");
  NSDictionary* dict = object;
  NSString* packaging = dict[@"packaging"];
  ET_CHECK_OR_RETURN_ERROR(
      [packaging isKindOfClass:NSString.class],
      InvalidProgram,
      "Missing Core AI packaging");
  ET_CHECK_OR_RETURN_ERROR(
      [packaging isEqualToString:@"inline"] ||
          [packaging isEqualToString:@"aot_compiled_inline"],
      NotSupported,
      "Unsupported Core AI packaging; expected .aimodel or AOT .aimodelc");
  Manifest manifest;
  manifest.aot_compiled = [packaging hasPrefix:@"aot_compiled_"];
  manifest.hash = dict[@"hash"];
  manifest.path = dict[@"path"];
  manifest.function = dict[@"function"];
  manifest.inputs = dict[@"input_names"];
  manifest.outputs = dict[@"output_names"];
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(manifest.hash) &&
          [manifest.hash
              rangeOfCharacterFromSet:[[NSCharacterSet
                                          characterSetWithCharactersInString:
                                              @"0123456789abcdef"] invertedSet]]
                  .location == NSNotFound,
      InvalidProgram,
      "Invalid Core AI model hash");
  NSMutableSet<NSString*>* bundles = [NSMutableSet set];
  if (manifest.aot_compiled) {
    manifest.platform = dict[@"platform"];
    manifest.archs = dict[@"archs"];
    ET_CHECK_OR_RETURN_ERROR(
        nonempty_string(manifest.platform),
        InvalidProgram,
        "Core AI AOT manifest requires a target platform; re-export the model");
    ET_CHECK_OR_RETURN_ERROR(
        [manifest.archs isKindOfClass:NSDictionary.class] &&
            manifest.archs.count > 0,
        InvalidProgram,
        "Core AI AOT manifest requires a nonempty architecture map; "
        "re-export the model");
    for (id arch in manifest.archs) {
      ET_CHECK_OR_RETURN_ERROR(
          relative_path(arch) && ![arch containsString:@"/"] &&
              [arch canBeConvertedToEncoding:NSUTF8StringEncoding],
          InvalidProgram,
          "Invalid Core AI AOT architecture name");
      NSString* bundle = [NSString stringWithFormat:@"model.%@.aimodelc", arch];
      id path = manifest.archs[arch];
      ET_CHECK_OR_RETURN_ERROR(
          relative_path(path) &&
              [path isEqualToString:[manifest.hash
                                        stringByAppendingPathComponent:bundle]],
          InvalidProgram,
          "Invalid Core AI AOT bundle path for architecture %s",
          [arch UTF8String]);
      [bundles addObject:bundle];
    }
    manifest.path = nil;
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        relative_path(manifest.path) &&
            [manifest.path
                isEqualToString:[manifest.hash
                                    stringByAppendingString:@"/model.aimodel"]],
        InvalidProgram,
        "Invalid Core AI source bundle path");
    [bundles addObject:@"model.aimodel"];
  }
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(manifest.function) && names_array(manifest.inputs) &&
          names_array(manifest.outputs),
      InvalidProgram,
      "Core AI manifest requires ordered input/output names; re-export the "
      "model");
  ET_CHECK_OR_RETURN_ERROR(
      parse_version(dict[@"min_deployment_version"], manifest.minimum_version),
      InvalidProgram,
      "Invalid Core AI minimum deployment version");
  manifest.files = dict[@"files"];
  ET_CHECK_OK_OR_RETURN_ERROR(validate_files(manifest.files, bundles));
  manifest.bundle_digests = dict[@"bundle_digests"];
  ET_CHECK_OR_RETURN_ERROR(
      [manifest.bundle_digests isKindOfClass:NSDictionary.class] &&
          [[NSSet setWithArray:manifest.bundle_digests.allKeys]
              isEqualToSet:bundles],
      InvalidProgram,
      "Core AI bundle digests must exactly cover declared bundles");
  for (NSString* bundle in bundles) {
    id digest = manifest.bundle_digests[bundle];
    ET_CHECK_OR_RETURN_ERROR(
        nonempty_string(digest) && [digest length] == 64 &&
            [digest
                rangeOfCharacterFromSet:
                    [[NSCharacterSet
                        characterSetWithCharactersInString:@"0123456789abcdef"]
                        invertedSet]]
                    .location == NSNotFound,
        InvalidProgram,
        "Invalid Core AI exported bundle digest");
  }
  return manifest;
}

Result<Manifest> select_assets(
    const Manifest& manifest,
    NSString* device_architecture,
    NSString* platform) {
  if (!manifest.aot_compiled) {
    return manifest;
  }
  ET_CHECK_OR_RETURN_ERROR(
      [manifest.platform isEqualToString:platform],
      DelegateInvalidCompatibility,
      "Core AI AOT target platform %s does not match runtime platform %s; "
      "re-export the model for this platform",
      manifest.platform.UTF8String,
      platform.UTF8String);
  NSArray<NSString*>* architectures = [manifest.archs.allKeys
      sortedArrayUsingComparator:^NSComparisonResult(NSString* a, NSString* b) {
        return [a compare:b options:NSLiteralSearch];
      }];
  NSString* available = [architectures componentsJoinedByString:@", "];
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(device_architecture),
      DelegateInvalidCompatibility,
      "Core AI SDK device architecture is unavailable (%s); available "
      "architectures: [%s]; re-export the model as .aimodel or for this device",
      device_architecture == nil ? "nil" : device_architecture.UTF8String,
      available.UTF8String);
  NSString* path = manifest.archs[device_architecture];
  ET_CHECK_OR_RETURN_ERROR(
      path != nil,
      DelegateInvalidCompatibility,
      "Core AI AOT has no bundle for device architecture %s; available "
      "architectures: [%s]; re-export the model for this device architecture",
      device_architecture.UTF8String,
      available.UTF8String);
  Manifest selected = manifest;
  selected.path = path;
  NSString* prefix = [path.lastPathComponent stringByAppendingString:@"/"];
  NSMutableDictionary<NSString*, NSNumber*>* files =
      [NSMutableDictionary dictionary];
  for (NSString* file in manifest.files) {
    if ([file hasPrefix:prefix]) {
      files[file] = manifest.files[file];
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      files.count > 0,
      InvalidProgram,
      "Core AI manifest has no files for selected device architecture %s; "
      "re-export the model",
      device_architecture.UTF8String);
  selected.files = files;
  selected.bundle_digests = @{
    path.lastPathComponent : manifest.bundle_digests[path.lastPathComponent]
  };
  return selected;
}

Result<NSURL*> prepare_source_bundle(
    const Manifest& manifest,
    const runtime::NamedDataMap* named_data,
    NSString* staging_root,
    NSString* key) {
  ET_CHECK_OR_RETURN_ERROR(
      relative_path(manifest.path),
      InvalidProgram,
      "Core AI bundle must be selected before preparing its source");
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(key) && key.length == 64 &&
          [key rangeOfCharacterFromSet:
                   [[NSCharacterSet characterSetWithCharactersInString:
                                        @"0123456789abcdef"] invertedSet]]
                  .location == NSNotFound,
      InvalidArgument,
      "Core AI staging key must be a 64-character lowercase hex digest");
  auto prepared = prepare_storage_root(staging_root, false);
  if (!prepared.ok()) {
    return prepared.error();
  }
  NSURL* root = [NSURL fileURLWithPath:prepared.get() isDirectory:YES];
  ET_CHECK_OK_OR_RETURN_ERROR(validate_files(
      manifest.files, [NSSet setWithObject:manifest.path.lastPathComponent]));
  NSDictionary<NSString*, NSNumber*>* files = manifest.files;
  NSURL* model_url = [[root URLByAppendingPathComponent:key isDirectory:YES]
      URLByAppendingPathComponent:manifest.path.lastPathComponent
                      isDirectory:YES];
  // Inspect, stage and publish relative to one root descriptor.
  FileDescriptor root_fd(retry_eintr([&] {
    return open(root.fileSystemRepresentation,
                O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      root_fd.get() >= 0, AccessFailed, "Cannot open Core AI storage root");
  auto state = inspect_entry(root_fd.get(), key, files);
  if (!state.ok()) {
    return state.error();
  }
  if (*state == EntryState::Valid) {
    return model_url;
  }
  // Recovery sources may still be in use by a previously acquired SDK model.
  ET_CHECK_OR_RETURN_ERROR(
      *state == EntryState::Missing,
      InvalidExternalData,
      "Cannot replace a damaged bookmark recovery source");
  ET_CHECK_OR_RETURN_ERROR(
      named_data != nullptr,
      InvalidProgram,
      "Missing NamedDataMap for Core AI source recovery");
  NSString* staging_name =
      [@".staging-" stringByAppendingString:NSUUID.UUID.UUIDString];
  ET_CHECK_OR_RETURN_ERROR(
      retry_eintr([&] {
        return mkdirat(
            root_fd.get(), staging_name.fileSystemRepresentation, 0700);
      }) == 0,
      AccessFailed,
      "Cannot create Core AI staging directory");
  StagingDirectory staging{[root URLByAppendingPathComponent:staging_name
                                                 isDirectory:YES]};
  FileDescriptor staging_fd(retry_eintr([&] {
    return openat(
        root_fd.get(),
        staging_name.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      staging_fd.get() >= 0,
      AccessFailed,
      "Cannot open Core AI staging directory");
  for (NSString* file in files) {
    @autoreleasepool {
      NSString* data_key =
          [NSString stringWithFormat:@"coreai/%@/%@", manifest.hash, file];
      auto buffer = named_data->get_data(data_key.UTF8String);
      ET_CHECK_OR_RETURN_ERROR(
          buffer.ok(),
          InvalidExternalData,
          "Missing selected Core AI named data: %s",
          data_key.UTF8String);
      ET_CHECK_OR_RETURN_ERROR(
          buffer->size() == files[file].unsignedLongLongValue &&
              (buffer->size() == 0 || buffer->data() != nullptr),
          InvalidExternalData,
          "Invalid Core AI named data buffer");
      ET_CHECK_OK_OR_RETURN_ERROR(write_storage_file(
          staging_fd.get(), file, buffer->data(), buffer->size()));
    }
  }
  ET_CHECK_OK_OR_RETURN_ERROR(validate_tree(staging_fd.get(), files));
  ET_CHECK_OR_RETURN_ERROR(
      storage_fault(StorageOperation::Rename) == 0,
      AccessFailed,
      "Core AI source publication interrupted");
  if (retry_eintr([&] {
        return renameatx_np(
            root_fd.get(),
            staging_name.fileSystemRepresentation,
            root_fd.get(),
            key.fileSystemRepresentation,
            RENAME_EXCL);
      }) == 0) {
    staging.url = nil;
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(root_fd.get()));
    return model_url;
  }
  ET_CHECK_OR_RETURN_ERROR(
      errno == EEXIST, AccessFailed, "Cannot publish Core AI stored bundle");
  // Another loader published first; use its source only if it is complete.
  auto winner = inspect_entry(root_fd.get(), key, files);
  if (!winner.ok()) {
    return winner.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      *winner == EntryState::Valid,
      InvalidExternalData,
      "Concurrently published Core AI source is incomplete");
  return model_url;
}

} // namespace executorch::backends::coreai
