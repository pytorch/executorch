/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::coreai {

class Manifest final {
 public:
  NSString* const export_hash;
  NSString* const function;
  NSArray<NSString*>* const inputs;
  NSArray<NSString*>* const outputs;
  NSDictionary<NSString*, NSNumber*>* const files;
  NSDictionary<NSString*, NSString*>* const bundle_digests;
  const bool aot_compiled;
  // Source bundles do not constrain the platform or architecture.
  NSString* const platform;
  NSDictionary<NSString*, NSString*>* const architectures;
  const NSOperatingSystemVersion minimum_version;

  Manifest(const Manifest&) = default;
  Manifest(Manifest&&) = default;
  Manifest& operator=(const Manifest&) = delete;
  Manifest& operator=(Manifest&&) = delete;

 private:
  Manifest(
      NSString* export_hash,
      NSString* function,
      NSArray<NSString*>* inputs,
      NSArray<NSString*>* outputs,
      NSDictionary<NSString*, NSNumber*>* files,
      NSDictionary<NSString*, NSString*>* bundle_digests,
      bool aot_compiled,
      NSString* platform,
      NSDictionary<NSString*, NSString*>* architectures,
      NSOperatingSystemVersion minimum_version);

  friend runtime::Result<Manifest> parse_manifest(NSData* data);
};

class SelectedBundle final {
 public:
  NSString* const export_hash;
  NSString* const path;
  NSString* const bundle_name;
  NSString* const digest;
  NSString* const function;
  NSArray<NSString*>* const inputs;
  NSArray<NSString*>* const outputs;
  NSDictionary<NSString*, NSNumber*>* const files;
  const bool aot_compiled;

  SelectedBundle(const SelectedBundle&) = default;
  SelectedBundle(SelectedBundle&&) = default;
  SelectedBundle& operator=(const SelectedBundle&) = delete;
  SelectedBundle& operator=(SelectedBundle&&) = delete;

 private:
  SelectedBundle(
      const Manifest& manifest,
      NSString* path,
      NSDictionary<NSString*, NSNumber*>* files);

  friend runtime::Result<SelectedBundle> select_bundle(
      const Manifest& manifest,
      NSString* device_architecture,
      NSString* platform);
};

runtime::Result<Manifest> parse_manifest(NSData* data);
runtime::Result<Manifest> parse_manifest(const void* data, size_t size);
runtime::Result<SelectedBundle> select_bundle(
    const Manifest& manifest,
    NSString* device_architecture,
    NSString* platform);

} // namespace executorch::backends::coreai
