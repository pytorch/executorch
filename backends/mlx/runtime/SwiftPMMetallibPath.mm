// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include "SwiftPMMetallibPath.h"

#include <filesystem>
#include <system_error>

namespace executorch::backends::mlx {
namespace {

constexpr const char* kResourceBundleName =
    "executorch_backend_mlx_resources.bundle";

const char* metallib_filename() {
#if TARGET_OS_SIMULATOR
  return "mlx-ios-simulator.metallib";
#elif TARGET_OS_IOS
  return "mlx-ios.metallib";
#elif TARGET_OS_OSX
  return "mlx-macos.metallib";
#else
  return nullptr;
#endif
}

std::optional<std::string> regular_file_path(NSURL* url) {
  if (url == nil || !url.fileURL || url.path == nil) {
    return std::nullopt;
  }

  std::error_code error;
  const std::filesystem::path path(url.fileSystemRepresentation);
  if (!std::filesystem::is_regular_file(path, error)) {
    return std::nullopt;
  }
  return path.string();
}

std::optional<std::string> find_in_resource_bundle(NSURL* bundle_url) {
  if (bundle_url == nil || !bundle_url.fileURL) {
    return std::nullopt;
  }

  NSBundle* bundle = [NSBundle bundleWithURL:bundle_url];
  if (bundle == nil) {
    return std::nullopt;
  }

  NSString* filename = [NSString stringWithUTF8String:metallib_filename()];
  if (filename == nil) {
    return std::nullopt;
  }

  if (auto path = regular_file_path(
          [bundle URLForResource:filename.stringByDeletingPathExtension
                   withExtension:filename.pathExtension])) {
    return path;
  }

  // SwiftPM's native build system can emit a flat resource bundle. Check the
  // bundle root explicitly in addition to Foundation's platform resource URL.
  return regular_file_path([bundle_url URLByAppendingPathComponent:filename]);
}

std::optional<std::string> find_from_container(NSURL* container_url) {
  if (container_url == nil || !container_url.fileURL) {
    return std::nullopt;
  }

  NSString* resource_bundle_name =
      [NSString stringWithUTF8String:kResourceBundleName];
  if ([container_url.lastPathComponent isEqualToString:resource_bundle_name]) {
    return find_in_resource_bundle(container_url);
  }

  NSBundle* container_bundle = [NSBundle bundleWithURL:container_url];
  if (container_bundle != nil) {
    NSURL* resource_bundle_url = [container_bundle
        URLForResource:resource_bundle_name.stringByDeletingPathExtension
         withExtension:resource_bundle_name.pathExtension];
    if (auto path = find_in_resource_bundle(resource_bundle_url)) {
      return path;
    }
  }

  return find_in_resource_bundle(
      [container_url URLByAppendingPathComponent:resource_bundle_name]);
}

void append_path(NSMutableOrderedSet<NSString*>* paths, NSURL* url) {
  if (url != nil && url.fileURL && url.path != nil) {
    [paths addObject:url.path];
  }
}

} // namespace

std::optional<std::string> find_swiftpm_metallib_path(
    const std::vector<std::string>& container_paths) {
  if (metallib_filename() == nullptr) {
    return std::nullopt;
  }

  @autoreleasepool {
    for (const auto& container_path : container_paths) {
      NSString* path = [NSString stringWithUTF8String:container_path.c_str()];
      if (path == nil) {
        continue;
      }
      if (auto metallib_path =
              find_from_container([NSURL fileURLWithPath:path])) {
        return metallib_path;
      }
    }
  }

  return std::nullopt;
}

std::optional<std::string> resolve_swiftpm_metallib_path() {
  @autoreleasepool {
    NSMutableOrderedSet<NSString*>* paths = [NSMutableOrderedSet orderedSet];
    NSBundle* main_bundle = NSBundle.mainBundle;
    append_path(paths, main_bundle.bundleURL);
    append_path(paths, main_bundle.resourceURL);

    for (NSBundle* bundle in NSBundle.allBundles) {
      append_path(paths, bundle.bundleURL);
      append_path(paths, bundle.resourceURL);
    }
    for (NSBundle* framework in NSBundle.allFrameworks) {
      append_path(paths, framework.bundleURL);
      append_path(paths, framework.resourceURL);
    }

    std::vector<std::string> container_paths;
    container_paths.reserve(paths.count);
    for (NSString* path in paths) {
      container_paths.emplace_back(path.fileSystemRepresentation);
    }
    return find_swiftpm_metallib_path(container_paths);
  }
}

} // namespace executorch::backends::mlx
