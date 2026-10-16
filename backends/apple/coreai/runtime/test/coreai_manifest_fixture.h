/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>
#include <string>
#import "coreai_assets.h"

namespace executorch::backends::coreai::testing {

// Synthetic identities keep host tests independent of the exporter hash.
inline NSString* fixture_identity(char digit) {
  EXPECT_TRUE((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'));
  return [NSString stringWithUTF8String:std::string(64, digit).c_str()];
}

inline NSMutableDictionary* manifest_dict() {
  return [@{
    @"packaging" : @"inline",
    @"hash" : @"ab",
    @"path" : @"ab/model.aimodel",
    @"function" : @"main",
    @"input_names" : @[ @"input_0" ],
    @"output_names" : @[ @"output_0" ],
    @"files" : @{@"model.aimodel/graph.bin" : @11},
    @"bundle_digests" : @{@"model.aimodel" : fixture_identity('1')},
    @"min_deployment_version" : @"26.0"
  } mutableCopy];
}

inline NSMutableDictionary* aot_manifest_dict() {
  auto dict = manifest_dict();
  [dict removeObjectForKey:@"path"];
  dict[@"packaging"] = @"aot_compiled_inline";
  dict[@"platform"] = @"macOS";
  dict[@"archs"] = @{
    @"arch_b" : @"ab/model.arch_b.aimodelc",
    @"arch_a" : @"ab/model.arch_a.aimodelc"
  };
  dict[@"files"] = @{
    @"model.arch_b.aimodelc/nested/weights.bin" : @7,
    @"model.arch_a.aimodelc/graph.bin" : @14,
    @"model.arch_b.aimodelc/graph.bin" : @14,
    @"model.arch_a.aimodelc/nested/weights.bin" : @7
  };
  dict[@"bundle_digests"] = @{
    @"model.arch_a.aimodelc" : fixture_identity('a'),
    @"model.arch_b.aimodelc" : fixture_identity('b')
  };
  return dict;
}

inline NSData* encode(NSDictionary* dict) {
  return [NSJSONSerialization dataWithJSONObject:dict options:0 error:nil];
}

} // namespace executorch::backends::coreai::testing
