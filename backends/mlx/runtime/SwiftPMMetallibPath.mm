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

void append_path(NSMutableOrderedSet<NSString*>* paths, NSURL* url) {
  if (url != nil && url.fileURL && url.path != nil) {
    [paths addObject:url.path];
  }
}

} // namespace

std::optional<std::string> find_swiftpm_metallib_path(
    const std::vector<std::string>& container_paths) {
  const char* filename = metallib_filename();
  if (filename == nullptr) {
    return std::nullopt;
  }

  for (const auto& container_path : container_paths) {
    const std::filesystem::path container(container_path);
    std::filesystem::path resource =
        container / kResourceBundleName / filename;
    std::error_code error;
    if (std::filesystem::is_regular_file(resource, error)) {
      return resource.string();
    }

    if (container.filename() == kResourceBundleName) {
      resource = container / filename;
      error.clear();
      if (std::filesystem::is_regular_file(resource, error)) {
        return resource.string();
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
