/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>
#include "coreai_manifest_fixture.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

TEST(CoreAIManifestTest, AcceptsInlineAndRejectsInvalidJson) {
  EXPECT_TRUE(parse_manifest(encode(manifest_dict())).ok());
  EXPECT_EQ(parse_manifest(nil).error(), Error::InvalidProgram);
  EXPECT_FALSE(
      parse_manifest([@"[]" dataUsingEncoding:NSUTF8StringEncoding]).ok());
  EXPECT_FALSE(
      parse_manifest([@"garbage" dataUsingEncoding:NSUTF8StringEncoding]).ok());
}

TEST(CoreAIManifestTest, RejectsUnknownPackaging) {
  auto dict = manifest_dict();
  dict[@"packaging"] = @"unknown";
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::NotSupported);
}

TEST(CoreAIManifestTest, RejectsUnsafeSourcePaths) {
  for (NSString* path in @[
         @"../model.aimodel",
         @"/ab/model.aimodel",
         @"ab/../model.aimodel",
         @"xx/model.aimodel"
       ]) {
    SCOPED_TRACE(path.UTF8String);
    auto dict = manifest_dict();
    dict[@"path"] = path;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, RejectsInvalidFileMaps) {
  for (id files in @[
         @[],
         @{},
         NSNull.null,
         @"files",
         @{@"model.aimodel/../escape" : @1},
         @{@"model.aimodel/bad\\name" : @1},
         @{@"/tmp/file" : @1},
         @{@"model.aimodel/one" : @1,
           @"model.aimodel/one/two" : @2}
       ]) {
    SCOPED_TRACE([files description].UTF8String);
    auto dict = manifest_dict();
    dict[@"files"] = files;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, RejectsInvalidInputNames) {
  for (id names in
       @[ @[ @"same", @"same" ], @[ @1 ], @[ @"" ], @"input", NSNull.null ]) {
    SCOPED_TRACE([names description].UTF8String);
    auto dict = manifest_dict();
    dict[@"input_names"] = names;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, ValidatesDeploymentFloor) {
  for (NSString* version in @[ @"", @"27.x", @"27.0.0.0" ]) {
    SCOPED_TRACE(version.UTF8String);
    auto dict = manifest_dict();
    dict[@"min_deployment_version"] = version;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
  auto dict = manifest_dict();
  dict[@"min_deployment_version"] = NSNull.null;
  EXPECT_TRUE(parse_manifest(encode(dict)).ok());
}

TEST(CoreAIManifestTest, ParsesDeploymentVersionComponents) {
  const struct {
    id value;
    NSOperatingSystemVersion expected;
  } rows[] = {
      {@"26", {26, 0, 0}},
      {@"26.4", {26, 4, 0}},
      {@"26.4.2", {26, 4, 2}},
      {NSNull.null, {0, 0, 0}}};
  for (const auto& row : rows) {
    SCOPED_TRACE([row.value description].UTF8String);
    auto dict = manifest_dict();
    dict[@"min_deployment_version"] = row.value;
    auto parsed = parse_manifest(encode(dict));
    ASSERT_TRUE(parsed.ok());
    EXPECT_EQ(parsed->minimum_version.majorVersion, row.expected.majorVersion);
    EXPECT_EQ(parsed->minimum_version.minorVersion, row.expected.minorVersion);
    EXPECT_EQ(parsed->minimum_version.patchVersion, row.expected.patchVersion);
  }
}

TEST(CoreAIManifestTest, RequiresMetadataMaps) {
  auto dict = manifest_dict();
  ASSERT_TRUE(parse_manifest(encode(dict)).ok());
  for (NSString* key in @[ @"files", @"bundle_digests" ]) {
    SCOPED_TRACE(key.UTF8String);
    NSMutableDictionary* invalid = [dict mutableCopy];
    [invalid removeObjectForKey:key];
    EXPECT_EQ(parse_manifest(encode(invalid)).error(), Error::InvalidProgram);
    invalid[key] = @[];
    EXPECT_EQ(parse_manifest(encode(invalid)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RejectsInvalidFileSizes) {
  for (id size in @[ @(-1), @0.5, @(std::numeric_limits<uint64_t>::max()) ]) {
    SCOPED_TRACE([size description].UTF8String);
    auto dict = manifest_dict();
    dict[@"files"] = @{@"model.aimodel/graph.bin" : size};
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, AcceptsIntegralFileSizes) {
  for (NSNumber* size in @[ @0, @11, @(std::numeric_limits<int64_t>::max()) ]) {
    SCOPED_TRACE(size.description.UTF8String);
    auto dict = manifest_dict();
    dict[@"files"] = @{@"model.aimodel/graph.bin" : size};
    EXPECT_TRUE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, RequiresExactLowercaseDigests) {
  for (NSString* digest in @[
         [fixture_identity('a') substringFromIndex:1],
         fixture_identity('a').uppercaseString,
         [@"g" stringByPaddingToLength:64 withString:@"g" startingAtIndex:0]
       ]) {
    SCOPED_TRACE(digest.UTF8String);
    auto dict = manifest_dict();
    dict[@"bundle_digests"] = @{@"model.aimodel" : digest};
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

class CoreAIManifestMetadataTest : public ::testing::TestWithParam<bool> {
 protected:
  NSMutableDictionary* manifest() {
    return GetParam() ? aot_manifest_dict() : manifest_dict();
  }
  NSString* file() {
    return GetParam() ? @"model.arch_b.aimodelc/graph.bin"
                      : @"model.aimodel/graph.bin";
  }
};

TEST_P(CoreAIManifestMetadataTest, ReturnsManifestFieldsAndNameOrder) {
  auto dict = manifest();
  dict[@"function"] = @"predict";
  dict[@"input_names"] = @[ @"input_1", @"input_0" ];
  dict[@"output_names"] = @[ @"output_2", @"output_0", @"output_1" ];
  dict[@"min_deployment_version"] = @"26.4.2";
  auto parsed = parse_manifest(encode(dict));
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed->aot_compiled, GetParam());
  EXPECT_TRUE([parsed->hash isEqual:dict[@"hash"]]);
  EXPECT_TRUE([parsed->function isEqual:dict[@"function"]]);
  EXPECT_TRUE([parsed->inputs isEqual:dict[@"input_names"]]);
  EXPECT_TRUE([parsed->outputs isEqual:dict[@"output_names"]]);
  EXPECT_TRUE([parsed->files isEqual:dict[@"files"]]);
  EXPECT_TRUE([parsed->bundle_digests isEqual:dict[@"bundle_digests"]]);
  EXPECT_EQ(parsed->minimum_version.majorVersion, 26);
  EXPECT_EQ(parsed->minimum_version.minorVersion, 4);
  EXPECT_EQ(parsed->minimum_version.patchVersion, 2);
  if (GetParam()) {
    EXPECT_EQ(parsed->path, nil);
    EXPECT_TRUE([parsed->platform isEqual:dict[@"platform"]]);
    EXPECT_TRUE([parsed->archs isEqual:dict[@"archs"]]);
  } else {
    EXPECT_TRUE([parsed->path isEqual:dict[@"path"]]);
    EXPECT_EQ(parsed->platform, nil);
    EXPECT_EQ(parsed->archs, nil);
  }
}

TEST_P(CoreAIManifestMetadataTest, RejectsFilesystemAliases) {
  NSString* composed = [NSString stringWithFormat:@"caf%C", 0x00e9];
  NSString* decomposed = composed.decomposedStringWithCanonicalMapping;
  for (NSArray<NSString*>* pair in @[
         @[ @"One", @"one" ],
         @[ @"One", @"one/two" ],
         @[ @"Dir/a", @"dir/b" ],
         @[ composed, decomposed ],
         @[ composed, [decomposed stringByAppendingPathComponent:@"child"] ],
         @[
           [composed stringByAppendingPathComponent:@"a"],
           [decomposed stringByAppendingPathComponent:@"b"]
         ],
         @[ composed.uppercaseString, decomposed ]
       ]) {
    SCOPED_TRACE(pair.description.UTF8String);
    auto dict = manifest();
    NSMutableDictionary* files = [dict[@"files"] mutableCopy];
    NSString* bundle = file().stringByDeletingLastPathComponent;
    files[[bundle stringByAppendingPathComponent:pair[0]]] = @1;
    files[[bundle stringByAppendingPathComponent:pair[1]]] = @2;
    dict[@"files"] = files;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST_P(CoreAIManifestMetadataTest, PreservesDistinctUnicodePaths) {
  auto dict = manifest();
  NSMutableDictionary* files = [dict[@"files"] mutableCopy];
  NSString* bundle = file().stringByDeletingLastPathComponent;
  NSString* composed = [NSString stringWithFormat:@"caf%C", 0x00e9];
  for (NSString* name in
       @[ composed, @"same/one", @"same/two", @"other/ONE" ]) {
    files[[bundle stringByAppendingPathComponent:name]] = @1;
  }
  dict[@"files"] = files;
  auto parsed = parse_manifest(encode(dict));
  ASSERT_TRUE(parsed.ok());
  EXPECT_TRUE([parsed->files isEqual:files]);
}

TEST_P(CoreAIManifestMetadataTest, RequiresExactBundleCoverage) {
  for (bool extra : {false, true}) {
    SCOPED_TRACE(extra ? "extra digest" : "missing digest");
    auto dict = manifest();
    NSMutableDictionary* digests = [dict[@"bundle_digests"] mutableCopy];
    if (extra) {
      digests[@"unexpected.aimodel"] = fixture_identity('e');
    } else {
      [digests removeObjectForKey:file().stringByDeletingLastPathComponent];
    }
    dict[@"bundle_digests"] = digests;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST_P(CoreAIManifestMetadataTest, RejectsInvalidHashes) {
  auto missing = manifest();
  [missing removeObjectForKey:@"hash"];
  EXPECT_EQ(parse_manifest(encode(missing)).error(), Error::InvalidProgram);
  for (id hash in @[ @"", @"g", @"AB", @"../x", @"a/b", @1, NSNull.null ]) {
    SCOPED_TRACE([hash description].UTF8String);
    auto dict = manifest();
    dict[@"hash"] = hash;
    if (GetParam()) {
      dict[@"archs"] = @{
        @"arch_a" :
            [NSString stringWithFormat:@"%@/model.arch_a.aimodelc", hash],
        @"arch_b" :
            [NSString stringWithFormat:@"%@/model.arch_b.aimodelc", hash]
      };
    } else {
      dict[@"path"] = [NSString stringWithFormat:@"%@/model.aimodel", hash];
    }
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST_P(CoreAIManifestMetadataTest, RejectsFilesForUndeclaredBundles) {
  auto dict = manifest();
  NSMutableDictionary* files = [dict[@"files"] mutableCopy];
  files[@"model.undeclared.aimodelc/graph.bin"] = @1;
  dict[@"files"] = files;
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
}

TEST_P(CoreAIManifestMetadataTest, RequiresFilesForEveryDeclaredBundle) {
  auto valid = manifest();
  for (NSString* bundle in valid[@"bundle_digests"]) {
    SCOPED_TRACE(bundle.UTF8String);
    NSMutableDictionary* dict = [valid mutableCopy];
    NSMutableDictionary* files = [dict[@"files"] mutableCopy];
    NSString* prefix = [bundle stringByAppendingString:@"/"];
    for (NSString* file in dict[@"files"]) {
      if ([file hasPrefix:prefix]) {
        [files removeObjectForKey:file];
      }
    }
    dict[@"files"] = files;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RequiresAotPlatform) {
  auto dict = aot_manifest_dict();
  [dict removeObjectForKey:@"platform"];
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  for (id platform in @[ @"", @1, @[], @{}, NSNull.null ]) {
    SCOPED_TRACE([platform description].UTF8String);
    dict[@"platform"] = platform;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RequiresNonemptyAotArchitectureMap) {
  auto dict = aot_manifest_dict();
  [dict removeObjectForKey:@"archs"];
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  for (id archs in @[ @{}, @[], @"arch_a", @1, NSNull.null ]) {
    SCOPED_TRACE([archs description].UTF8String);
    dict[@"archs"] = archs;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RejectsUnsafeAotArchitectureNames) {
  for (NSString* arch in @[ @"", @".", @"..", @"../arch_a", @"a/b", @"a\\b" ]) {
    SCOPED_TRACE(arch.UTF8String);
    auto dict = aot_manifest_dict();
    NSString* bundle = [NSString stringWithFormat:@"model.%@.aimodelc", arch];
    dict[@"archs"] = @{arch : [@"ab/" stringByAppendingString:bundle]};
    dict[@"files"] = @{[bundle stringByAppendingString:@"/graph.bin"] : @1};
    dict[@"bundle_digests"] = @{bundle : fixture_identity('a')};
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RejectsAliasedAotBundles) {
  NSString* composed = [NSString stringWithFormat:@"arch_%C", 0x00e9];
  for (NSArray<NSString*>* pair in @[
         @[ @"arch_a", @"ARCH_A" ],
         @[ composed, composed.decomposedStringWithCanonicalMapping ]
       ]) {
    SCOPED_TRACE(pair.description.UTF8String);
    auto dict = aot_manifest_dict();
    NSMutableDictionary* archs = [NSMutableDictionary dictionary];
    NSMutableDictionary* files = [NSMutableDictionary dictionary];
    NSMutableDictionary* digests = [NSMutableDictionary dictionary];
    for (NSString* arch in pair) {
      NSString* bundle = [NSString stringWithFormat:@"model.%@.aimodelc", arch];
      archs[arch] = [@"ab" stringByAppendingPathComponent:bundle];
      files[[bundle stringByAppendingPathComponent:@"graph.bin"]] = @1;
      digests[bundle] = fixture_identity('a');
    }
    ASSERT_EQ(archs.count, 2);
    dict[@"archs"] = archs;
    dict[@"files"] = files;
    dict[@"bundle_digests"] = digests;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, RejectsInvalidAotBundlePaths) {
  for (id path in @[
         @"",
         @"../model.arch_a.aimodelc",
         @"/ab/model.arch_a.aimodelc",
         @"ab/../model.arch_a.aimodelc",
         @"xx/model.arch_a.aimodelc",
         @"ab/model.arch_b.aimodelc",
         @"ab/model.aimodel",
         @1,
         NSNull.null
       ]) {
    SCOPED_TRACE([path description].UTF8String);
    auto dict = aot_manifest_dict();
    NSMutableDictionary* archs = [dict[@"archs"] mutableCopy];
    archs[@"arch_a"] = path;
    dict[@"archs"] = archs;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Delivery,
    CoreAIManifestMetadataTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Aot" : "Source";
    });

TEST(CoreAIManifestTest, SelectsAotArchitectureWithoutMutatingManifest) {
  auto dict = aot_manifest_dict();
  auto parsed = parse_manifest(encode(dict));
  ASSERT_TRUE(parsed.ok());
  ASSERT_TRUE(parsed->aot_compiled);
  EXPECT_EQ(parsed->path, nil);
  for (NSString* arch in @[ @"arch_a", @"arch_b" ]) {
    SCOPED_TRACE(arch.UTF8String);
    auto selected = select_assets(parsed.get(), arch, @"macOS");
    ASSERT_TRUE(selected.ok());
    EXPECT_TRUE([selected->path isEqualToString:dict[@"archs"][arch]]);
    EXPECT_EQ(selected->files.count, 2);
    NSString* bundle = selected->path.lastPathComponent;
    EXPECT_TRUE(([selected->bundle_digests
        isEqual:@{bundle : dict[@"bundle_digests"][bundle]}]));
    EXPECT_TRUE(([selected->files isEqual:@{
      [bundle stringByAppendingPathComponent:@"graph.bin"] : @14,
      [bundle stringByAppendingPathComponent:@"nested/weights.bin"] : @7
    }]));
    EXPECT_EQ(parsed->path, nil);
  }
}

TEST(CoreAIManifestTest, RejectsMismatchedAotPlatformAndArchitecture) {
  auto ios = aot_manifest_dict();
  ios[@"platform"] = @"iOS";
  auto parsed_ios = parse_manifest(encode(ios));
  ASSERT_TRUE(parsed_ios.ok());
  EXPECT_TRUE(select_assets(parsed_ios.get(), @"arch_b", @"iOS").ok());
  auto parsed = parse_manifest(encode(aot_manifest_dict()));
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(
      select_assets(parsed.get(), @"arch_b", @"iOS").error(),
      Error::DelegateInvalidCompatibility);
  EXPECT_EQ(
      select_assets(parsed.get(), @"arch_b", nil).error(),
      Error::DelegateInvalidCompatibility);
  for (id arch in @[ @"missing", @"ARCH_B", @"", NSNull.null ]) {
    SCOPED_TRACE([arch description].UTF8String);
    EXPECT_EQ(
        select_assets(parsed.get(), arch == NSNull.null ? nil : arch, @"macOS")
            .error(),
        Error::DelegateInvalidCompatibility);
  }
}
