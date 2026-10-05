/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispatch/dispatch.h>
#include <pthread.h>
#include "coreai_bookmarks.h"
#include "coreai_fault_scope.h"
#include "coreai_source_fixture.h"

namespace executorch::backends::coreai::testing {
using runtime::Error;
using runtime::Result;

namespace {
Result<Manifest> source_manifest(bool aot = false) {
  auto parsed = parse_manifest(encode(aot ? aot_manifest_dict() : manifest_dict()));
  if (!parsed.ok()) return parsed.error();
  return select_assets(parsed.get(), @"arch_b", @"macOS");
}
}  // namespace

TEST_F(CoreAISourceTest, RejectsUnselectedAotSource) {
  auto parsed = parse_manifest(encode(aot_manifest_dict()));
  ASSERT_TRUE(parsed.ok());
  ASSERT_TRUE(parsed->aot_compiled);
  ASSERT_EQ(parsed->path, nil);
  EXPECT_FALSE(prepare_source_bundle(parsed.get(), nullptr, nil).ok());
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
    auto selected = select_assets(parsed.get(), arch, @"macOS");
    ASSERT_TRUE(selected.ok());
    auto first = prepare_source_bundle(selected.get(), &data, storage.url.path);
    ASSERT_TRUE(first.ok());
    EXPECT_EQ(data.attempts, 2);
    EXPECT_EQ(data.requests, 2);
    NSURL* url = first.get();
    EXPECT_TRUE([url.lastPathComponent isEqualToString:selected->path.lastPathComponent]);
    NSData* contents = [NSData dataWithContentsOfURL:[url URLByAppendingPathComponent:@"graph.bin"]];
    ASSERT_TRUE([contents isEqual:[NSData dataWithBytes:"compiled\0graph" length:14]]);

    NSString* other_arch = [arch isEqualToString:@"arch_a"] ? @"arch_b" : @"arch_a";
    NSString* other_key =
        [NSString stringWithFormat:@"coreai/ab/model.%@.aimodelc/graph.bin", other_arch];
    data.files[other_key.UTF8String] = "must not affect identity";
    auto reused = prepare_source_bundle(selected.get(), &data, storage.url.path);
    ASSERT_TRUE(reused.ok());
    EXPECT_TRUE([reused.get() isEqual:url]);
    EXPECT_EQ(data.attempts, 2);

    auto unselected_changed = aot_manifest_dict();
    NSMutableDictionary* digests = [unselected_changed[@"bundle_digests"] mutableCopy];
    digests[[NSString stringWithFormat:@"model.%@.aimodelc", other_arch]] = fixture_identity('f');
    unselected_changed[@"bundle_digests"] = digests;
    NSMutableDictionary* files = [unselected_changed[@"files"] mutableCopy];
    files[[NSString stringWithFormat:@"model.%@.aimodelc/extra", other_arch]] = @0;
    unselected_changed[@"files"] = files;
    auto changed_parsed = parse_manifest(encode(unselected_changed));
    ASSERT_TRUE(changed_parsed.ok());
    auto unchanged = select_assets(changed_parsed.get(), arch, @"macOS");
    ASSERT_TRUE(unchanged.ok());
    auto unchanged_source = prepare_source_bundle(unchanged.get(), &data, storage.url.path);
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
      const Manifest* manifest = nullptr;
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
          &threads[launched], nullptr,
          [](void* context) -> void* {
            auto& worker = *static_cast<Worker*>(context);
            @autoreleasepool {
              worker.started =
                  dispatch_semaphore_wait(worker.start,
                                          dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0;
              if (!worker.started) return nullptr;
              auto assets =
                  worker.fixture->prepare_source_bundle(*worker.manifest, worker.data, worker.root);
              worker.error = assets.error();
              if (assets.ok()) worker.path = assets.get().path.UTF8String;
            }
            return nullptr;
          },
          &worker);
      if (launch_error != 0) break;
    }
    for (size_t i = 0; i < launched; ++i) dispatch_semaphore_signal(start);
    int join_errors[4] = {};
    for (size_t i = 0; i < launched; ++i) join_errors[i] = pthread_join(threads[i], nullptr);
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
    EXPECT_EQ([NSFileManager.defaultManager contentsOfDirectoryAtPath:storage.url.path error:nil]
                  .count,
              1u);
  }
}

TEST_F(CoreAISourceTest, ConcurrentColdAndWarmSource) {
  check_concurrent_source();
}

TEST_F(CoreAISourceTest, RejectsInvalidKeysAndMissingSourceInputs) {
  auto manifest = source_manifest();
  ASSERT_TRUE(manifest.ok());
  NSString* valid_key = fixture_identity('a');
  for (NSString* key in @[ @"../escape", [valid_key substringToIndex:63] ]) {
    SCOPED_TRACE(key.UTF8String);
    TestDirectory invalid_storage;
    ASSERT_NE(invalid_storage.url, nil);
    TestData invalid_data;
    EXPECT_EQ(executorch::backends::coreai::prepare_source_bundle(
                  manifest.get(), &invalid_data, invalid_storage.url.path, key)
                  .error(),
              Error::InvalidArgument);
    EXPECT_EQ(invalid_data.attempts, 0);
  }

  TestData data;
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  EXPECT_FALSE(prepare_source_bundle(manifest.get(), nullptr, storage.url.path).ok());
  NSURL* absent = [storage.url URLByAppendingPathComponent:@"absent"];
  EXPECT_FALSE(prepare_source_bundle(manifest.get(), &data, absent.path).ok());
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:absent.path]);
  auto recovered = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(
      [NSData dataWithContentsOfURL:[recovered.get() URLByAppendingPathComponent:@"graph.bin"]]
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
    auto assets = prepare_source_bundle(manifest.get(), &data, storage.url.path);
    ASSERT_TRUE(assets.ok());
    EXPECT_EQ(data.releases, data.requests);
    first_url = assets.get();
    EXPECT_EQ(
        [NSData dataWithContentsOfURL:[first_url URLByAppendingPathComponent:@"graph.bin"]].length,
        11u);
  }
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:first_url.path]);
  auto reused = prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first_url]);
}

TEST_F(CoreAISourceTest, PersistentIdentityTracksDigestNotFileOrder) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto dict = manifest_dict();
  dict[@"files"] = @{@"model.aimodel/graph.bin" : @11, @"model.aimodel/nested/weights.bin" : @7};
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "weights";
  auto manifest = parse_manifest(encode(dict));
  ASSERT_TRUE(manifest.ok());
  auto first = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(first.ok());
  NSURL* first_url = first.get();
  dict[@"files"] = @{@"model.aimodel/nested/weights.bin" : @7, @"model.aimodel/graph.bin" : @11};
  auto reordered_manifest = parse_manifest(encode(dict));
  ASSERT_TRUE(reordered_manifest.ok());
  auto reordered = prepare_source_bundle(reordered_manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(reordered.ok());
  EXPECT_TRUE([reordered.get() isEqual:first_url]);
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "changed";
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('3')};
  auto changed_manifest = parse_manifest(encode(dict));
  ASSERT_TRUE(changed_manifest.ok());
  auto changed = prepare_source_bundle(changed_manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(changed.ok());
  EXPECT_FALSE([changed.get() isEqual:first_url]);
  EXPECT_TRUE(
      [[NSData dataWithContentsOfURL:[first_url URLByAppendingPathComponent:@"nested/weights.bin"]]
          isEqual:[@"weights" dataUsingEncoding:NSUTF8StringEncoding]]);
  EXPECT_EQ(data.releases, data.requests);
}

TEST_F(CoreAISourceTest, PartialMaterializationCleansStagingAndCanRetry) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto dict = manifest_dict();
  dict[@"files"] = @{@"model.aimodel/graph.bin" : @11, @"model.aimodel/nested/weights.bin" : @7};
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
  data.files["coreai/ab/model.aimodel/nested/weights.bin"] = "weights";
  auto manifest = parse_manifest(encode(dict));
  ASSERT_TRUE(manifest.ok());
  data.fail_at = 2;
  EXPECT_FALSE(prepare_source_bundle(manifest.get(), &data, storage.url.path).ok());
  EXPECT_EQ(
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:storage.url.path error:nil].count,
      0u);
  EXPECT_EQ(data.releases, data.requests);
  data.fail_at = 0;
  EXPECT_TRUE(prepare_source_bundle(manifest.get(), &data, storage.url.path).ok());
  EXPECT_EQ(
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:storage.url.path error:nil].count,
      1u);
}

TEST_F(CoreAISourceTest, NestedEmptyAndLargeFilesMaterializeAndReuseWithoutNamedData) {
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
  data.files["coreai/ab/model.aimodel/nested/deeper/large.bin"] = std::string(large_size, 'x');
  auto manifest = parse_manifest(encode(dict));
  ASSERT_TRUE(manifest.ok());
  auto first = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(data.attempts, 3);
  EXPECT_EQ(data.requests, data.releases);
  NSData* empty = [NSData
      dataWithContentsOfURL:[first.get() URLByAppendingPathComponent:@"nested/empty"]];
  ASSERT_NE(empty, nil);
  EXPECT_EQ(empty.length, 0u);
  NSData* large = [NSData
      dataWithContentsOfURL:[first.get() URLByAppendingPathComponent:@"nested/deeper/large.bin"]];
  ASSERT_NE(large, nil);
  EXPECT_EQ(large.length, large_size);

  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(storage.url, snapshot));
  const int attempts = data.attempts;
  auto reused = prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first.get()]);
  EXPECT_EQ(data.attempts, attempts);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(data.metadata_requests, 0);
  EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
}

TEST_F(CoreAISourceTest, ExplicitStagingKeyReusesSourceWithoutWritesOrRepublication) {
  BookmarkDirectory directory;
  ASSERT_NE(directory.url, nil);
  TestData data;
  auto manifest = source_manifest();
  ASSERT_TRUE(manifest.ok());
  auto root = resolve_bookmark_root(directory.url.path);
  ASSERT_TRUE(root.ok());
  NSString* key = fixture_identity('9');
  auto lock = lock_bookmark(root.get(), key);
  ASSERT_TRUE(lock.ok());
  auto staging = prepare_bookmark_staging(*lock.get());
  ASSERT_TRUE(staging.ok());
  auto first = executorch::backends::coreai::prepare_source_bundle(manifest.get(), &data,
                                                                   staging.get(), key);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(data.attempts, 1);
  EXPECT_TRUE(([first.get().path
      isEqualToString:[root.get()
                          stringByAppendingPathComponent:
                              [NSString stringWithFormat:@"staging/%@/model.aimodel", key]]]));
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(directory.url, snapshot));
  const int attempts = data.attempts;
  const int requests = data.requests;
  data.files.clear();
  auto reused = executorch::backends::coreai::prepare_source_bundle(manifest.get(), &data,
                                                                    staging.get(), key);
  ASSERT_TRUE(reused.ok());
  EXPECT_TRUE([reused.get() isEqual:first.get()]);
  EXPECT_EQ(data.attempts, attempts);
  EXPECT_EQ(data.requests, requests);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(data.metadata_requests, 0);
  EXPECT_TRUE(snapshot_matches(directory.url, snapshot));
}

TEST_F(CoreAISourceTest, ExclusivePublicationValidatesWinnerAndRemovesLosingStaging) {
  TestDirectory storage;
  TestDirectory winner_storage;
  ASSERT_NE(storage.url, nil);
  ASSERT_NE(winner_storage.url, nil);
  TestData data;
  TestData winner_data;
  auto manifest = source_manifest();
  ASSERT_TRUE(manifest.ok());
  auto winner = prepare_source_bundle(manifest.get(), &winner_data, winner_storage.url.path);
  ASSERT_TRUE(winner.ok());
  NSURL* winner_entry = winner.get().URLByDeletingLastPathComponent;
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(winner_entry, snapshot));
  NSURL* root = storage.url;
  NSURL* final =
      [root URLByAppendingPathComponent:manifest->bundle_digests[manifest->path.lastPathComponent]];
  __block int publications = 0;
  __block bool callback_ok = true;
  {
    StorageFaultScope fault(StorageOperation::Rename);
    fault.armed = false;
    fault.observe = ^(StorageOperation operation) {
      if (operation != StorageOperation::Rename) return;
      ++publications;
      callback_ok &= staging_directories(root).count == 1;
      callback_ok &= [NSFileManager.defaultManager moveItemAtURL:winner_entry toURL:final error:nil];
    };
    auto result = prepare_source_bundle(manifest.get(), &data, storage.url.path);
    EXPECT_TRUE(callback_ok);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(
        [result.get() isEqual:[final URLByAppendingPathComponent:manifest->path.lastPathComponent]]);
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(fault.hits, 0);
  }
  EXPECT_EQ(data.attempts, manifest->files.count);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(staging_directories(storage.url).count, 0u);
  EXPECT_TRUE(snapshot_matches(final, snapshot));
}

}  // namespace executorch::backends::coreai::testing
