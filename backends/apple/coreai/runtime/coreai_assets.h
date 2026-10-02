/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::coreai {

struct Manifest {
  NSString* hash = nil;
  NSString* path = nil;
  NSString* function = nil;
  NSArray<NSString*>* inputs = nil;
  NSArray<NSString*>* outputs = nil;
  NSDictionary<NSString*, NSNumber*>* files = nil;
  NSDictionary<NSString*, NSString*>* bundle_digests = nil;
  bool aot_compiled = false;
  NSString* platform = nil;
  NSDictionary<NSString*, NSString*>* archs = nil;
  NSOperatingSystemVersion minimum_version{};
};

runtime::Result<Manifest> parse_manifest(NSData* data);
// Resolve an already validated manifest before reading or materializing assets.
runtime::Result<Manifest> select_assets(
    const Manifest& manifest,
    NSString* device_architecture,
    NSString* platform);
// Caller holds the key's disk lock and has prepared staging_root.
// Inline bundles are atomically published at staging_root/key/bundle.
runtime::Result<NSURL*> prepare_source_bundle(
    const Manifest& manifest,
    const runtime::NamedDataMap* named_data,
    NSString* staging_root,
    NSString* key);

} // namespace executorch::backends::coreai
