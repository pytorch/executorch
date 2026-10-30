/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "../coreai_cache_key.h"

#include <gtest/gtest.h>
#include <limits.h>
#include "coreai_bookmark_fixture.h"
#include "coreai_manifest_fixture.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {

NSString* key_for_device(NSString* device) {
  NSString* model = [@"" stringByPaddingToLength:64
                                      withString:@"a"
                                 startingAtIndex:0];
  NSString* options = [@"" stringByPaddingToLength:64
                                        withString:@"b"
                                   startingAtIndex:0];
  return [NSString stringWithFormat:@"%@-%@-%@", model, device, options];
}

TEST(CoreAICacheKeyIdentityTest, IgnoresBindingsAndDeliveryLocation) {
  auto baseline = bookmark_bundle();
  ASSERT_TRUE(baseline.ok());
  auto expected = test_cache_key(baseline.get());
  ASSERT_TRUE(expected.ok());

  auto dictionary = manifest_dict();
  dictionary[@"export_hash"] = @"cd";
  dictionary[@"path"] = @"cd/model.aimodel";
  dictionary[@"function"] = @"another_function";
  dictionary[@"input_names"] = @[ @"different_input" ];
  dictionary[@"output_names"] = @[ @"different_output" ];
  auto changed = selected_bundle(dictionary);
  ASSERT_TRUE(changed.ok());
  auto actual = test_cache_key(changed.get());
  ASSERT_TRUE(actual.ok());
  EXPECT_TRUE([actual.get() isEqual:expected.get()]);
}

TEST(CoreAICacheKeyIdentityTest, SeparatesExportedBundleIdentity) {
  auto baseline = bookmark_bundle();
  ASSERT_TRUE(baseline.ok());
  auto expected = test_cache_key(baseline.get());
  ASSERT_TRUE(expected.ok());

  auto digest_changed = manifest_dict();
  digest_changed[@"bundle_digests"] =
      @{@"model.aimodel" : fixture_identity('2')};
  auto reweighted = selected_bundle(digest_changed);
  ASSERT_TRUE(reweighted.ok());

  auto aot = bookmark_bundle(true);
  ASSERT_TRUE(aot.ok());
  for (const SelectedBundle* bundle : {&reweighted.get(), &aot.get()}) {
    auto actual = test_cache_key(*bundle);
    ASSERT_TRUE(actual.ok());
    EXPECT_FALSE([actual.get() isEqual:expected.get()]);
  }
}

TEST(CoreAICacheKeyIdentityTest, SeparatesBundleNameWithSameDigestAndMode) {
  auto dictionary = aot_manifest_dict();
  dictionary[@"bundle_digests"] = @{
    @"model.arch_a.aimodelc" : fixture_identity('b'),
    @"model.arch_b.aimodelc" : fixture_identity('b')
  };
  auto manifest = parse_manifest(encode(dictionary));
  ASSERT_TRUE(manifest.ok());
  auto first = select_bundle(manifest.get(), @"arch_a", @"macOS");
  auto second = select_bundle(manifest.get(), @"arch_b", @"macOS");
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  auto first_key = bundle_identity(first.get());
  auto second_key = bundle_identity(second.get());
  ASSERT_TRUE(first_key.ok());
  ASSERT_TRUE(second_key.ok());
  EXPECT_FALSE([first_key.get() isEqual:second_key.get()]);
}

TEST(CoreAICacheKeyIdentityTest, RejectsUnknownBundleName) {
  auto manifest = parse_manifest(encode(manifest_dict()));
  ASSERT_TRUE(manifest.ok());
  EXPECT_EQ(
      bundle_identity(manifest.get(), @"unknown.aimodel").error(),
      Error::InvalidArgument);
}

TEST(CoreAICacheKeyIdentityTest, SeparatesDeviceAndSpecializationIdentity) {
  auto bundle = bookmark_bundle();
  ASSERT_TRUE(bundle.ok());
  auto baseline = test_cache_key(bundle.get());
  auto platform =
      cache_variant_key(bundle.get(), @"iOS", @"arch_b", @"default");
  auto architecture =
      cache_variant_key(bundle.get(), @"macOS", @"arch_a", @"default");
  auto specialization = cache_variant_key(
      bundle.get(),
      @"macOS",
      @"arch_b",
      @"{\"version\":1,\"compute\":[\"cpu\"]}");
  ASSERT_TRUE(baseline.ok());
  for (auto* changed : {&platform, &architecture, &specialization}) {
    ASSERT_TRUE(changed->ok());
    EXPECT_FALSE([changed->get() isEqual:baseline.get()]);
    EXPECT_TRUE([cache_key_bundle_identity(changed->get())
        isEqual:cache_key_bundle_identity(baseline.get())]);
  }
}

TEST(CoreAICacheKeyIdentityTest, RejectsIncompletePlainIdentity) {
  auto bundle = bookmark_bundle();
  ASSERT_TRUE(bundle.ok());
  EXPECT_FALSE(
      cache_variant_key(bundle.get(), nil, @"arch_b", @"default").ok());
  EXPECT_FALSE(cache_variant_key(bundle.get(), @"macOS", @"", @"default").ok());
  EXPECT_FALSE(cache_variant_key(bundle.get(), @"macOS", @"arch_b", nil).ok());
  EXPECT_FALSE(
      cache_variant_key(bundle.get(), @"mac_OS", @"arch_b", @"default").ok());
}

TEST(CoreAICacheKeyFormatTest, AcceptsReadableDeviceComponents) {
  for (NSString* device in
       @[ @"macOS_h17p", @"iOS_arch_b", @"visionOS_h17p-high" ]) {
    NSString* key = key_for_device(device);
    EXPECT_TRUE(is_cache_key(key));
    EXPECT_TRUE(
        [cache_key_bundle_identity(key) isEqual:[key substringToIndex:64]]);
  }
}

TEST(CoreAICacheKeyFormatTest, RejectsUnsafeOrIncompleteDevices) {
  for (NSString* device in @[
         @"",
         @"iOS",
         @"iOS_",
         @"_h17p",
         @"iOS_../h17p",
         @"iOS_h17p\\extra",
         @"iOS_h17p extra",
         @"iOS_h\u00e9",
         @"iOS_h17p.bookmark"
       ]) {
    NSString* key = key_for_device(device);
    EXPECT_FALSE(is_cache_key(key));
    EXPECT_EQ(cache_key_bundle_identity(key), nil);
  }
}

TEST(CoreAICacheKeyFormatTest, LeavesRoomForBookmarkExtension) {
  constexpr NSUInteger max_device_length = NAME_MAX - 130 - 9;
  NSString* device = [@"iOS_" stringByPaddingToLength:max_device_length
                                           withString:@"a"
                                      startingAtIndex:0];
  NSString* key = key_for_device(device);
  ASSERT_EQ(key.length + 9, NAME_MAX);
  EXPECT_TRUE(is_cache_key(key));
  EXPECT_FALSE(
      is_cache_key(key_for_device([device stringByAppendingString:@"a"])));
}

TEST(CoreAICacheKeyFormatTest, RequiresExactHashBoundaries) {
  NSString* key = key_for_device(@"iOS_h17p");
  EXPECT_FALSE(is_cache_key(nil));
  EXPECT_FALSE(is_cache_key(@""));
  EXPECT_FALSE(is_cache_key([key substringToIndex:64]));
  EXPECT_FALSE(is_cache_key([key substringFromIndex:1]));
  EXPECT_FALSE(is_cache_key([key stringByAppendingString:@"0"]));
  EXPECT_FALSE(is_cache_key(key.uppercaseString));
  EXPECT_FALSE(is_cache_key([key stringByAppendingString:@".bookmark"]));
  EXPECT_FALSE(
      is_cache_key([key stringByReplacingCharactersInRange:NSMakeRange(64, 1)
                                                withString:@"_"]));
}

TEST(CoreAICacheKeyFormatTest, LockStripesStayWithinConfiguredPool) {
  auto bundle = bookmark_bundle();
  ASSERT_TRUE(bundle.ok());
  NSMutableSet<NSString*>* names = [NSMutableSet set];
  for (NSUInteger index = 0; index < 4096; ++index) {
    auto key = test_cache_key(
        bundle.get(), [NSString stringWithFormat:@"identity-%lu", index]);
    ASSERT_TRUE(key.ok());
    auto name = cache_entry_lock_name(key.get());
    ASSERT_TRUE(name.ok());
    ASSERT_TRUE([name.get() hasPrefix:@"stripe-v1-"]);
    ASSERT_TRUE([name.get() hasSuffix:@".lock"]);
    NSString* encoded =
        [name.get() substringWithRange:NSMakeRange(10, name.get().length - 15)];
    unsigned int stripe = 0;
    ASSERT_TRUE([[NSScanner scannerWithString:encoded] scanHexInt:&stripe]);
    EXPECT_LT(stripe, kCacheEntryLockStripeCount);
    [names addObject:name.get()];
  }
  EXPECT_GT(names.count, 1u);
  EXPECT_LE(names.count, kCacheEntryLockStripeCount);
}

} // namespace
