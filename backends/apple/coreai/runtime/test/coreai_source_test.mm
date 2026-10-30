/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include "coreai_cache_entry.h"
#include "coreai_cache_key.h"
#include "coreai_fault_scope.h"
#include "coreai_source_fixture.h"
#include "coreai_test_utils.h"

namespace executorch::backends::coreai::testing {
using runtime::Error;
using runtime::Result;

TEST_F(CoreAISourceTest, UnicodePathsMaterializeAndReuse) {
  NSString* composed = [NSString stringWithFormat:@"caf%C", 0x00e9];
  for (NSString* name in
       @[ composed, composed.decomposedStringWithCanonicalMapping ]) {
    SCOPED_TRACE(name.UTF8String);
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    TestData data;
    data.files.clear();
    NSString* relative = [NSString stringWithFormat:@"%@/%@.bin", name, name];
    NSString* file = [@"model.aimodel" stringByAppendingPathComponent:relative];
    auto dict = manifest_dict();
    dict[@"files"] = @{file : @7};
    data.files[[NSString stringWithFormat:@"coreai/ab/%@", file].UTF8String] =
        "unicode";
    auto manifest = selected_bundle(dict);
    ASSERT_TRUE(manifest.ok());
    auto source =
        prepare_source_bundle(manifest.get(), &data, storage.url.path);
    ASSERT_TRUE(source.ok());
    NSData* bytes = [NSData
        dataWithContentsOfURL:[source.get()
                                  URLByAppendingPathComponent:relative]];
    EXPECT_TRUE([bytes isEqual:[NSData dataWithBytes:"unicode" length:7]]);
    auto reused =
        prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
    ASSERT_TRUE(reused.ok());
    EXPECT_TRUE([reused.get() isEqual:source.get()]);
    EXPECT_EQ(data.attempts, 1);
    EXPECT_EQ(data.requests, data.releases);
  }
}

TEST_F(CoreAISourceTest, RejectsAliasedPathsBeforeReadingAssets) {
  NSString* composed = [NSString stringWithFormat:@"caf%C", 0x00e9];
  NSString* decomposed = composed.decomposedStringWithCanonicalMapping;
  for (NSArray<NSString*>* pair in @[
         @[ @"A.bin", @"a.bin" ],
         @[ @"Dir/x", @"dir/y" ],
         @[ @"Dir", @"dir/x" ],
         @[ composed, decomposed ],
         @[
           [composed stringByAppendingPathComponent:@"x"],
           [decomposed stringByAppendingPathComponent:@"y"]
         ]
       ]) {
    SCOPED_TRACE(pair.description.UTF8String);
    TestData data;
    auto dictionary = manifest_dict();
    dictionary[@"files"] = @{
      [@"model.aimodel" stringByAppendingPathComponent:pair[0]] : @1,
      [@"model.aimodel" stringByAppendingPathComponent:pair[1]] : @1
    };
    EXPECT_EQ(
        parse_manifest(encode(dictionary)).error(), Error::InvalidProgram);
    EXPECT_EQ(data.attempts, 0);
  }
}

TEST_F(CoreAISourceTest, AotMaterializesOnlySelectedArchitecture) {
  auto parsed = parse_manifest(encode(aot_manifest_dict()));
  ASSERT_TRUE(parsed.ok());
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  for (NSString* arch in @[ @"arch_a", @"arch_b" ]) {
    SCOPED_TRACE(arch.UTF8String);
    TestData data;
    aot_data(data, arch);
    auto selected = select_bundle(parsed.get(), arch, @"macOS");
    ASSERT_TRUE(selected.ok());
    auto first = prepare_source_bundle(selected.get(), &data, storage.url.path);
    ASSERT_TRUE(first.ok());
    EXPECT_EQ(data.attempts, 2);
    EXPECT_EQ(data.requests, 2);
    NSURL* url = first.get();
    EXPECT_TRUE([url.lastPathComponent
        isEqualToString:selected->path.lastPathComponent]);
    NSData* contents = [NSData
        dataWithContentsOfURL:[url URLByAppendingPathComponent:@"graph.bin"]];
    ASSERT_TRUE([contents isEqual:[NSData dataWithBytes:"compiled\0graph"
                                                 length:14]]);

    NSString* other_arch =
        [arch isEqualToString:@"arch_a"] ? @"arch_b" : @"arch_a";
    NSString* other_key = [NSString
        stringWithFormat:@"coreai/ab/model.%@.aimodelc/graph.bin", other_arch];
    data.files[other_key.UTF8String] = "must not affect identity";
    auto reused =
        prepare_source_bundle(selected.get(), &data, storage.url.path);
    ASSERT_TRUE(reused.ok());
    EXPECT_TRUE([reused.get() isEqual:url]);
    EXPECT_EQ(data.attempts, 2);

    auto unselected_changed = aot_manifest_dict();
    NSMutableDictionary* digests =
        [unselected_changed[@"bundle_digests"] mutableCopy];
    digests[[NSString stringWithFormat:@"model.%@.aimodelc", other_arch]] =
        fixture_identity('f');
    unselected_changed[@"bundle_digests"] = digests;
    NSMutableDictionary* files = [unselected_changed[@"files"] mutableCopy];
    files[[NSString stringWithFormat:@"model.%@.aimodelc/extra", other_arch]] =
        @0;
    unselected_changed[@"files"] = files;
    auto changed_parsed = parse_manifest(encode(unselected_changed));
    ASSERT_TRUE(changed_parsed.ok());
    auto unchanged = select_bundle(changed_parsed.get(), arch, @"macOS");
    ASSERT_TRUE(unchanged.ok());
    auto unchanged_source =
        prepare_source_bundle(unchanged.get(), &data, storage.url.path);
    ASSERT_TRUE(unchanged_source.ok());
    EXPECT_TRUE([unchanged_source.get() isEqual:url]);
    EXPECT_EQ(data.attempts, 2);
    EXPECT_EQ(data.releases, data.requests);
  }
}

void CoreAISourceTest::check_concurrent_source() {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto selected = source_manifest();
  ASSERT_TRUE(selected.ok());
  for (int pass = 0; pass < 2; ++pass) {
    SCOPED_TRACE(pass == 0 ? "cold" : "warm");
    using FixturePointer = decltype(this);
    struct Worker {
      FixturePointer fixture = nullptr;
      const SelectedBundle* manifest = nullptr;
      TestData* data = nullptr;
      NSString* root = nil;
      dispatch_semaphore_t start = nullptr;
      bool started = false;
      Error error = Error::Internal;
      std::string path;
    } workers[4];
    pthread_t threads[4];
    dispatch_semaphore_t start = dispatch_semaphore_create(0);
    int launch_error = 0;
    size_t launched = 0;
    for (; launched < 4; ++launched) {
      auto& worker = workers[launched];
      worker.fixture = this;
      worker.manifest = &selected.get();
      worker.data = &data;
      worker.root = storage.url.path;
      worker.start = start;
      launch_error = pthread_create(
          &threads[launched],
          nullptr,
          [](void* context) -> void* {
            auto& worker = *static_cast<Worker*>(context);
            @autoreleasepool {
              worker.started =
                  dispatch_semaphore_wait(worker.start, test_deadline()) == 0;
              if (!worker.started)
                return nullptr;
              auto assets = worker.fixture->prepare_source_bundle(
                  *worker.manifest, worker.data, worker.root);
              worker.error = assets.error();
              if (assets.ok())
                worker.path = assets.get().path.UTF8String;
            }
            return nullptr;
          },
          &worker);
      if (launch_error != 0)
        break;
    }
    for (size_t i = 0; i < launched; ++i)
      dispatch_semaphore_signal(start);
    int join_errors[4] = {};
    for (size_t i = 0; i < launched; ++i)
      join_errors[i] = pthread_join(threads[i], nullptr);
    ASSERT_EQ(launch_error, 0);
    for (size_t i = 0; i < 4; ++i) {
      SCOPED_TRACE(i);
      ASSERT_EQ(join_errors[i], 0);
      ASSERT_TRUE(workers[i].started);
      ASSERT_EQ(workers[i].error, Error::Ok);
      EXPECT_EQ(workers[i].path, workers[0].path);
    }
    EXPECT_EQ(data.attempts, 1);
    EXPECT_EQ(data.requests, 1);
    EXPECT_EQ(data.requests, data.releases);
    EXPECT_EQ(data.metadata_requests, 0);
    EXPECT_EQ(
        [NSFileManager.defaultManager
            contentsOfDirectoryAtPath:staging_root(storage.url).path
                                error:nil]
            .count,
        1u);
  }
}

TEST_F(CoreAISourceTest, PerKeyLockSerializesColdAndWarmPreparation) {
  check_concurrent_source();
}

TEST_F(CoreAISourceTest, MissingSourceRequiresNamedData) {
  auto bundle = source_manifest();
  ASSERT_TRUE(bundle.ok());
  TestData data;
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  EXPECT_EQ(
      prepare_source_bundle(bundle.get(), nullptr, storage.url.path).error(),
      Error::InvalidProgram);
  auto recovered = prepare_source_bundle(bundle.get(), &data, storage.url.path);
  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(
      [NSData
          dataWithContentsOfURL:[recovered.get()
                                    URLByAppendingPathComponent:@"graph.bin"]]
          .length,
      11u);
  EXPECT_EQ(data.requests, data.releases);
}

TEST_F(CoreAISourceTest, PublishedSourceOutlivesResult) {
  TestData data;
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  auto manifest = source_manifest();
  ASSERT_TRUE(manifest.ok());
  NSURL* first_url = nil;
  {
    auto assets =
        prepare_source_bundle(manifest.get(), &data, storage.url.path);
    ASSERT_TRUE(assets.ok());
    EXPECT_EQ(data.releases, data.requests);
    first_url = assets.get();
    EXPECT_EQ(
        [NSData
            dataWithContentsOfURL:[first_url
                                      URLByAppendingPathComponent:@"graph.bin"]]
            .length,
        11u);
  }
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:first_url.path]);
  auto reused =
      prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first_url]);
}

TEST_F(CoreAISourceTest, PersistentIdentityTracksDigestNotFileOrder) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto dict = manifest_dict();
  dict[@"files"] = @{
    @"model.aimodel/graph.bin" : @11,
    @"model.aimodel/nested/weights.bin" : @7
  };
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "weights";
  auto manifest = selected_bundle(dict);
  ASSERT_TRUE(manifest.ok());
  auto first = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(first.ok());
  NSURL* first_url = first.get();
  dict[@"files"] = @{
    @"model.aimodel/nested/weights.bin" : @7,
    @"model.aimodel/graph.bin" : @11
  };
  auto reordered_manifest = selected_bundle(dict);
  ASSERT_TRUE(reordered_manifest.ok());
  auto reordered =
      prepare_source_bundle(reordered_manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(reordered.ok());
  EXPECT_TRUE([reordered.get() isEqual:first_url]);
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "changed";
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('3')};
  auto changed_manifest = selected_bundle(dict);
  ASSERT_TRUE(changed_manifest.ok());
  auto changed =
      prepare_source_bundle(changed_manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(changed.ok());
  EXPECT_FALSE([changed.get() isEqual:first_url]);
  EXPECT_TRUE(
      [[NSData dataWithContentsOfURL:[first_url URLByAppendingPathComponent:
                                                    @"nested/weights.bin"]]
          isEqual:[@"weights" dataUsingEncoding:NSUTF8StringEncoding]]);
  EXPECT_EQ(data.releases, data.requests);
}

TEST_F(CoreAISourceTest, PartialMaterializationCleansStagingAndCanRetry) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto dict = manifest_dict();
  dict[@"files"] = @{
    @"model.aimodel/graph.bin" : @11,
    @"model.aimodel/nested/weights.bin" : @7
  };
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "weights";
  auto manifest = selected_bundle(dict);
  ASSERT_TRUE(manifest.ok());
  data.fail_at = 2;
  EXPECT_EQ(
      prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
      Error::AccessFailed);
  EXPECT_EQ(data.attempts, 2);
  EXPECT_EQ(
      [NSFileManager.defaultManager
          contentsOfDirectoryAtPath:staging_root(storage.url).path
                              error:nil]
          .count,
      0u);
  EXPECT_EQ(data.releases, data.requests);
  data.fail_at = 0;
  EXPECT_TRUE(
      prepare_source_bundle(manifest.get(), &data, storage.url.path).ok());
  EXPECT_EQ(
      [NSFileManager.defaultManager
          contentsOfDirectoryAtPath:staging_root(storage.url).path
                              error:nil]
          .count,
      1u);
}

TEST_F(
    CoreAISourceTest,
    NestedEmptyAndLargeFilesMaterializeAndReuseWithoutNamedData) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto dict = manifest_dict();
  constexpr size_t large_size = 2 * 1024 * 1024 + 17;
  dict[@"files"] = @{
    @"model.aimodel/graph.bin" : @11,
    @"model.aimodel/nested/empty" : @0,
    @"model.aimodel/nested/deeper/large.bin" : @(large_size)
  };
  data.files["coreai/ab/model.aimodel/nested/empty"] = "";
  data.files["coreai/ab/model.aimodel/nested/deeper/large.bin"] =
      std::string(large_size, 'x');
  auto manifest = selected_bundle(dict);
  ASSERT_TRUE(manifest.ok());
  auto first = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(data.attempts, 3);
  EXPECT_EQ(data.requests, data.releases);
  NSData* empty = [NSData
      dataWithContentsOfURL:[first.get()
                                URLByAppendingPathComponent:@"nested/empty"]];
  ASSERT_NE(empty, nil);
  EXPECT_EQ(empty.length, 0u);
  NSData* large = [NSData
      dataWithContentsOfURL:
          [first.get() URLByAppendingPathComponent:@"nested/deeper/large.bin"]];
  ASSERT_NE(large, nil);
  EXPECT_EQ(large.length, large_size);

  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(storage.url, snapshot));
  const int attempts = data.attempts;
  auto reused =
      prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first.get()]);
  EXPECT_EQ(data.attempts, attempts);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(data.metadata_requests, 0);
  EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
}

TEST_F(
    CoreAISourceTest,
    ExplicitStagingKeyReusesSourceWithoutWritesOrRepublication) {
  TestDirectory directory;
  ASSERT_NE(directory.url, nil);
  TestData data;
  auto bundle = source_manifest();
  ASSERT_TRUE(bundle.ok());
  auto key = cache_variant_key(
      bundle.get(), @"macOS", @"arch_b", @"explicit-test-options");
  ASSERT_TRUE(key.ok());
  auto lock = acquire_test_cache_entry(directory.url.path, key.get());
  ASSERT_TRUE(lock.ok());
  auto first = executorch::backends::coreai::prepare_source_bundle(
      bundle.get(), &data, *lock.get());
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(data.attempts, 1);
  NSString* expected_path = [directory.url.URLByResolvingSymlinksInPath.path
      stringByAppendingPathComponent:
          [NSString stringWithFormat:@"staging/%@/model.aimodel", key.get()]];
  EXPECT_TRUE([first.get().path isEqualToString:expected_path]);
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(directory.url, snapshot));
  const int attempts = data.attempts;
  const int requests = data.requests;
  data.files.clear();
  auto reused = executorch::backends::coreai::prepare_source_bundle(
      bundle.get(), &data, *lock.get());
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first.get()]);
  EXPECT_EQ(data.attempts, attempts);
  EXPECT_EQ(data.requests, requests);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(data.metadata_requests, 0);
  EXPECT_TRUE(snapshot_matches(directory.url, snapshot));
}

} // namespace executorch::backends::coreai::testing
