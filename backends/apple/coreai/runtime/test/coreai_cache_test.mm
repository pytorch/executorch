/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_cache.h"
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <cerrno>
#include <cstdlib>
#include <vector>
#include "coreai_bookmark_fixture.h"
#include "coreai_fake_loader.h"
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_pte_fixture.h"
#include "coreai_source_fixture.h"

namespace executorch::backends::coreai::testing {
namespace {

using runtime::Error;
using SegmentType = runtime::DataLoader::SegmentInfo::Type;
using ::testing::AssertionFailure;
using ::testing::AssertionResult;
using ::testing::AssertionSuccess;

class CoreAICacheTest : public ::testing::Test {
 protected:
  void SetUp() override { runtime::runtime_init(); }

  void drain_callbacks() {
    auto drained = bridge.wait_for_callbacks(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
    if (!drained) {
      ADD_FAILURE() << drained.message();
      // Timed-out callbacks may still borrow this test's PTE or observations.
      std::_Exit(EXIT_FAILURE);
    }
  }

  ScopedFakeBridgeState bridge;
};

struct CacheEntry {
  NSString* key = nil;
  NSData* bookmark = nil;
  NSURL* stage = nil;
};

AssertionResult seed_cache(NSURL* root, NSDictionary* dict, BookmarkFakeLoader* loader,
                           CacheEntry& output, bool bookmark = true, bool stage = true,
                           NSString* architecture = @"arch_b") {
  if (root == nil || loader == nil) {
    return AssertionFailure() << "Cache seed requires a root and fake loader";
  }
  auto parsed = parse_manifest(encode(dict));
  if (!parsed.ok()) {
    return AssertionFailure() << "Cannot parse cache seed manifest";
  }
  auto selected = select_assets(parsed.get(), architecture, @"macOS");
  if (!selected.ok()) {
    return AssertionFailure() << "Cannot select cache seed assets";
  }
  auto key = bookmark_key(selected.get(), @"macOS", architecture);
  if (!key.ok()) {
    return AssertionFailure() << "Cannot compute cache seed key";
  }
  CacheEntry entry{
      key.get(), nil,
      [[root URLByAppendingPathComponent:@"staging"] URLByAppendingPathComponent:key.get()]};
  auto lock = lock_bookmark(root.path, entry.key);
  if (!lock.ok()) {
    return AssertionFailure() << "Cannot lock cache seed bookmark";
  }
  if (stage) {
    auto staging = prepare_bookmark_staging(*lock.get());
    if (!staging.ok()) {
      return AssertionFailure() << "Cannot prepare cache seed staging";
    }
    TestData data;
    if (selected->aot_compiled) {
      aot_data(data, architecture);
    }
    auto staged = executorch::backends::coreai::prepare_source_bundle(selected.get(), &data,
                                                                      staging.get(), entry.key);
    if (data.requests != data.releases) {
      return AssertionFailure() << "Cache seed source buffers were not released";
    }
    if (!staged.ok()) {
      return AssertionFailure() << "Cannot stage cache seed source bundle";
    }
  }
  if (bookmark) {
    entry.bookmark = [NSUUID.UUID.UUIDString dataUsingEncoding:NSUTF8StringEncoding];
    BookmarkFakeModel* model = [[BookmarkFakeModel alloc] init];
    if (model == nil || entry.bookmark == nil) {
      return AssertionFailure() << "Cannot create cache seed model/bookmark";
    }
    model.owner = loader;
    model.bookmark = entry.bookmark;
    loader.models[entry.bookmark] = model;
    if (write_bookmark(*lock.get(), entry.bookmark) != Error::Ok) {
      return AssertionFailure() << "Cannot write cache seed bookmark";
    }
  }
  output = entry;
  return AssertionSuccess();
}

bool cache_exists(NSURL* url) { return [NSFileManager.defaultManager fileExistsAtPath:url.path]; }

void cache_entry_removed(NSURL* root, const CacheEntry& entry) {
  EXPECT_FALSE(cache_exists(test_bookmark_url(root, entry.key)));
  EXPECT_FALSE(cache_exists(entry.stage));
  EXPECT_TRUE(cache_exists([[root URLByAppendingPathComponent:@"locks"]
      URLByAppendingPathComponent:[entry.key stringByAppendingString:@".lock"]]));
}

NSMutableDictionary* cache_manifest(char identity) {
  auto dict = manifest_dict();
  dict[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity(identity)};
  return dict;
}

TEST_F(CoreAICacheTest, EmptyAndMissingRootsAreNoncreatingNoops) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    BookmarkLoaderScope fake;
    ASSERT_NE(fake.loader, nil);
    FBSyntheticPTE empty({{"forward", {}}});
    NSDictionary* snapshot = nil;
    ASSERT_TRUE(asset_tree_snapshot(root.url, snapshot));
    const int queries = bridge.state().architecture_queries;
    bridge.state().available = false;
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    EXPECT_EQ(clear_cache_for_pte(empty, root.url.path.UTF8String), Error::Ok);
    NSURL* missing = [root.url URLByAppendingPathComponent:@"absent/nested/models"];
    EXPECT_EQ(clear_cache(missing.path.UTF8String), Error::Ok);
    EXPECT_EQ(clear_cache_for_pte(empty, missing.path.UTF8String), Error::Ok);
    EXPECT_FALSE(cache_exists(missing));
    EXPECT_TRUE(snapshot_matches(root.url, snapshot));
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    EXPECT_EQ(fake.loader->restores.load(), 0);
    EXPECT_EQ(fake.loader->specializations.load(), 0);
    EXPECT_EQ(bridge.state().architecture_queries.load(), queries);
    EXPECT_TRUE(empty.check_released());
  }
}

TEST_F(CoreAICacheTest, ValidatesExplicitRootsAndPtePaths) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    BookmarkLoaderScope fake;
    ASSERT_NE(fake.loader, nil);
    FBSyntheticPTE empty({{"forward", {}}});
    EXPECT_EQ(clear_cache("relative"), Error::InvalidArgument);
    EXPECT_EQ(clear_cache_for_pte(empty, "relative"), Error::InvalidArgument);
    EXPECT_EQ(clear_cache(nullptr), Error::InvalidArgument);
    EXPECT_EQ(clear_cache_for_pte(static_cast<const char*>(nullptr), root.url.path.UTF8String),
              Error::InvalidArgument);
    EXPECT_TRUE(empty.check_released());
  }
}

TEST_F(CoreAICacheTest, SelectsAllMethodsDeduplicatesAndEvictsInKeyOrder) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkDirectory isolated;
    BookmarkLoaderScope fake;
    auto source_dict = manifest_dict();
    auto second_dict = cache_manifest('2');
    auto aot_dict = aot_manifest_dict();
    CacheEntry source, second, aot, unselected, unrelated, other_root;
    ASSERT_TRUE(seed_cache(root.url, source_dict, fake.loader, source));
    ASSERT_TRUE(seed_cache(root.url, second_dict, fake.loader, second));
    ASSERT_TRUE(seed_cache(root.url, aot_dict, fake.loader, aot));
    ASSERT_TRUE(seed_cache(root.url, aot_dict, fake.loader, unselected, true, true, @"arch_a"));
    ASSERT_TRUE(seed_cache(root.url, cache_manifest('3'), fake.loader, unrelated));
    ASSERT_TRUE(seed_cache(isolated.url, source_dict, fake.loader, other_root));
    NSDictionary* isolation = nil;
    NSDictionary* unselected_stage = nil;
    NSDictionary* unrelated_stage = nil;
    ASSERT_TRUE(asset_tree_snapshot(isolated.url, isolation));
    ASSERT_TRUE(asset_tree_snapshot(unselected.stage, unselected_stage));
    ASSERT_TRUE(asset_tree_snapshot(unrelated.stage, unrelated_stage));
    FBSyntheticPTE pte(
        {{"empty", {}},
         {"first",
          {{"UnregisteredBackend", [@"not JSON" dataUsingEncoding:NSUTF8StringEncoding], true},
           {"CoreAIBackend", encode(source_dict)},
           {"CoreAIBackend", encode(second_dict), true}}},
         {"last",
          {{"CoreAIBackend", encode(source_dict), true},
           {"CoreAIBackend", encode(aot_manifest_dict()), true}}}});
    NSMutableArray<NSData*>* evictions = [NSMutableArray array];
    fake.loader.onEvict = ^(NSData* bookmark) {
      [evictions addObject:bookmark];
    };
    const int queries = bridge.state().architecture_queries;
    EXPECT_EQ(clear_cache_for_pte(pte, root.url.path.UTF8String), Error::Ok);
    drain_callbacks();
    EXPECT_EQ(bridge.state().architecture_queries.load(), queries + 1);
    EXPECT_EQ(fake.loader->evictions.load(), 3);
    EXPECT_EQ(fake.loader->specializations.load(), 0);
    EXPECT_EQ(fake.loader->bindings.load(), 0);
    EXPECT_EQ(fake.loader->restores.load(), 0);
    NSArray<NSString*>* ordered =
        [@[ source.key, second.key, aot.key ] sortedArrayUsingSelector:@selector(compare:)];
    NSDictionary<NSString*, NSData*>* bookmarks =
        @{source.key : source.bookmark, second.key : second.bookmark, aot.key : aot.bookmark};
    ASSERT_EQ(evictions.count, ordered.count);
    for (NSUInteger i = 0; i < ordered.count; ++i) {
      SCOPED_TRACE(::testing::Message() << "sorted eviction index " << i);
      EXPECT_TRUE([evictions[i] isEqual:bookmarks[ordered[i]]]);
    }
    cache_entry_removed(root.url, source);
    cache_entry_removed(root.url, second);
    cache_entry_removed(root.url, aot);
    EXPECT_NE(fake.loader.models[other_root.bookmark], nil);
    EXPECT_NE(fake.loader.models[unselected.bookmark], nil);
    EXPECT_TRUE(snapshot_matches(isolated.url, isolation));
    EXPECT_TRUE(snapshot_matches(unselected.stage, unselected_stage));
    EXPECT_TRUE(snapshot_matches(unrelated.stage, unrelated_stage));
    EXPECT_EQ(pte.backend_indices, (std::vector<size_t>{1, 2, 3}));
    EXPECT_EQ(pte.backend_descriptors,
              (std::vector<std::string>{"CoreAIBackend", "CoreAIBackend", "CoreAIBackend"}));
    EXPECT_TRUE(pte.check_released());
  }
}

TEST_F(CoreAICacheTest, FilePathOverloadClearsReferencedEntry) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    TestDirectory distribution;
    ASSERT_NE(distribution.url, nil);
    CacheEntry entry;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, entry));
    FBSyntheticPTE pte({{"forward", {{"CoreAIBackend", encode(manifest_dict())}}}});
    NSURL* path = [distribution.url URLByAppendingPathComponent:@"model.pte"];
    ASSERT_TRUE([[NSData dataWithBytes:pte.bytes.data() length:pte.bytes.size()] writeToURL:path
                                                                                    options:0
                                                                                      error:nil]);
    EXPECT_EQ(clear_cache_for_pte(path.path.UTF8String, root.url.path.UTF8String), Error::Ok);
    drain_callbacks();
    EXPECT_EQ(fake.loader->evictions.load(), 1);
    cache_entry_removed(root.url, entry);
  }
}

TEST_F(CoreAICacheTest, RejectsInvalidLaterDelegatesBeforeMutation) {
  @autoreleasepool {
    const char* rows[] = {"malformed manifest", "wrong AOT platform", "backend load failure"};
    for (int row = 0; row < 3; ++row) {
      SCOPED_TRACE(rows[row]);
      BookmarkDirectory root;
      BookmarkLoaderScope fake;
      CacheEntry source;
      ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, source));
      NSDictionary* snapshot = nil;
      ASSERT_TRUE(asset_tree_snapshot(root.url, snapshot));
      CacheDelegateSpec later{"CoreAIBackend", encode(aot_manifest_dict()), true};
      if (row == 0) later.data = [@"malformed" dataUsingEncoding:NSUTF8StringEncoding];
      if (row == 1) {
        auto dict = aot_manifest_dict();
        dict[@"platform"] = @"iOS";
        later.data = encode(dict);
      }
      FBSyntheticPTE pte(
          {{"first", {{"CoreAIBackend", encode(manifest_dict())}}}, {"later", {later}}});
      if (row == 2) pte.fail_backend_index = 0;
      NSDictionary* models = [fake.loader.models copy];
      const auto error = clear_cache_for_pte(pte, root.url.path.UTF8String);
      drain_callbacks();
      EXPECT_NE(error, Error::Ok);
      if (row == 2) EXPECT_EQ(error, Error::AccessFailed);
      EXPECT_EQ(fake.loader->evictions.load(), 0);
      EXPECT_EQ(fake.loader->specializations.load(), 0);
      EXPECT_EQ(fake.loader->bindings.load(), 0);
      EXPECT_EQ(fake.loader->restores.load(), 0);
      EXPECT_TRUE(snapshot_matches(root.url, snapshot));
      EXPECT_TRUE([fake.loader.models isEqual:models]);
      EXPECT_TRUE(pte.check_released());
    }
  }
}

TEST_F(CoreAICacheTest, ProcessedBuffersRemainAliveThroughEviction) {
  @autoreleasepool {
    BookmarkDirectory root;
    CacheEntry source;
    NSData* manifest = encode(manifest_dict());
    FBSyntheticPTE valid({{"forward",
                           {{"CoreAIBackend", manifest, true},
                            {"CoreAIBackend", manifest, true},
                            {"CoreAIBackend", manifest}}}});
    // Destroy the loader scope before the PTE captured by its callback.
    BookmarkLoaderScope fake;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, source));
    __block int observations = 0;
    __block bool program_alive = false;
    __block int backend_releases = -1;
    FBSyntheticPTE* observed = &valid;
    fake.loader.onEvict = ^(NSData*) {
      ++observations;
      program_alive = observed->program_loads.load() == observed->program_releases.load() + 1;
      backend_releases = observed->backend_releases.load();
    };
    EXPECT_EQ(clear_cache_for_pte(valid, root.url.path.UTF8String), Error::Ok);
    drain_callbacks();
    EXPECT_EQ(observations, 1);
    EXPECT_TRUE(program_alive);
    EXPECT_EQ(backend_releases, 0);
    EXPECT_EQ(valid.backend_loads.load(), 2);
    EXPECT_EQ(valid.backend_indices, (std::vector<size_t>{0, 1}));
    EXPECT_TRUE(valid.program_alive_at_backend_release);
    EXPECT_TRUE(valid.check_released());
    ASSERT_FALSE(valid.release_order.empty());
    EXPECT_EQ(valid.release_order.back(), SegmentType::Program);
  }
}

TEST_F(CoreAICacheTest, InventoriesBookmarkAndStagingUnionPreservingUnownedFiles) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    CacheEntry both, bookmark_only, stage_only;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, both));
    ASSERT_TRUE(seed_cache(root.url, cache_manifest('2'), fake.loader, bookmark_only, true, false));
    ASSERT_TRUE(seed_cache(root.url, cache_manifest('3'), fake.loader, stage_only, false, true));
    NSURL* temporary = [root.url URLByAppendingPathComponent:@"staging/.staging-unkeyed"];
    ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:temporary
                                       withIntermediateDirectories:NO
                                                        attributes:nil
                                                             error:nil]);
    NSURL* untouched = [temporary URLByAppendingPathComponent:@"partial"];
    ASSERT_TRUE([[@"do not infer ownership" dataUsingEncoding:NSUTF8StringEncoding]
        writeToURL:untouched
           options:0
             error:nil]);
    NSURL* external_sentinel = [root.url URLByAppendingPathComponent:@"caller.pte"];
    ASSERT_TRUE([[@"PTE" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:external_sentinel
                                                                    options:0
                                                                      error:nil]);
    NSDictionary* temp_snapshot = nil;
    NSDictionary* locks = nil;
    ASSERT_TRUE(asset_tree_snapshot(temporary, temp_snapshot));
    ASSERT_TRUE(asset_tree_snapshot([root.url URLByAppendingPathComponent:@"locks"], locks));
    const int queries = bridge.state().architecture_queries;
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    EXPECT_EQ(fake.loader->evictions.load(), 2);
    EXPECT_EQ(fake.loader->restores.load(), 0);
    EXPECT_EQ(fake.loader->specializations.load(), 0);
    EXPECT_EQ(bridge.state().architecture_queries.load(), queries);
    cache_entry_removed(root.url, both);
    cache_entry_removed(root.url, bookmark_only);
    cache_entry_removed(root.url, stage_only);
    EXPECT_TRUE(snapshot_matches([root.url URLByAppendingPathComponent:@"locks"], locks));
    EXPECT_TRUE(snapshot_matches(temporary, temp_snapshot));
    EXPECT_TRUE(cache_exists(external_sentinel));
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    EXPECT_EQ(fake.loader->evictions.load(), 2);
  }
}

TEST_F(CoreAICacheTest, RemovesStagingOnlyEntriesWithoutSdkAvailability) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    CacheEntry stage_only;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, stage_only, false, true));
    bridge.state().available = false;
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    cache_entry_removed(root.url, stage_only);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    EXPECT_EQ(fake.loader->restores.load(), 0);
  }
}

TEST_F(CoreAICacheTest, NilLoaderFactoryPreservesEntriesAndAllowsRetry) {
  @autoreleasepool {
    ASSERT_TRUE(bridge.state().available);
    ASSERT_FALSE(bridge.state().fail_loader_factory);
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    CacheEntry entry;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, entry));
    NSDictionary* snapshot = nil;
    ASSERT_TRUE(asset_tree_snapshot(root.url, snapshot));
    bridge.state().fail_loader_factory = true;
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Internal);
    bridge.state().fail_loader_factory = false;
    EXPECT_TRUE(snapshot_matches(root.url, snapshot));
    EXPECT_NE(fake.loader.models[entry.bookmark], nil);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    EXPECT_EQ(fake.loader->restores.load(), 0);
    EXPECT_EQ(fake.loader->specializations.load(), 0);
    EXPECT_EQ(fake.loader->bindings.load(), 0);
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    cache_entry_removed(root.url, entry);
  }
}

TEST_F(CoreAICacheTest, UnavailableSdkPreservesTrackedEntriesAndAllowsRetry) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    CacheEntry entry;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, entry));
    NSDictionary* snapshot = nil;
    ASSERT_TRUE(asset_tree_snapshot(root.url, snapshot));
    bridge.state().available = false;
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::NotSupported);
    bridge.state().available = true;
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    EXPECT_TRUE(snapshot_matches(root.url, snapshot));
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
    cache_entry_removed(root.url, entry);
  }
}

TEST_F(CoreAICacheTest, SourceRemovalFailuresKeepBookmarkForRetry) {
  @autoreleasepool {
    const struct {
      int match_index;
      bool bookmark_removal;
      const char* name;
    } rows[] = {{2, false, "staged-tree removal"}, {6, true, "bookmark removal"}};
    for (const auto& row : rows) {
      SCOPED_TRACE(row.name);
      BookmarkDirectory root;
      BookmarkLoaderScope fake;
      CacheEntry entry;
      ASSERT_TRUE(seed_cache(root.url, aot_manifest_dict(), fake.loader, entry));
      {
        StorageFaultScope fault(StorageOperation::Remove);
        fault.match_index = row.match_index;
        EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::AccessFailed);
        EXPECT_EQ(fault.hits, 1);
      }
      drain_callbacks();
      auto saved = saved_bookmark(root.url, entry.key);
      ASSERT_TRUE(saved.ok());
      EXPECT_TRUE([saved.get() isEqual:entry.bookmark]);
      EXPECT_EQ(cache_exists(entry.stage), !row.bookmark_removal);
      if (fake.loader.models[entry.bookmark] == nil) {
        fake.loader.evictError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                     code:ENOENT
                                                 userInfo:nil];
      }
      EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::Ok);
      cache_entry_removed(root.url, entry);
    }
  }
}

TEST_F(CoreAICacheTest, IndependentKeysContinueInSortedOrderAfterFailures) {
  @autoreleasepool {
    BookmarkDirectory root;
    BookmarkLoaderScope fake;
    CacheEntry first, second, third;
    ASSERT_TRUE(seed_cache(root.url, manifest_dict(), fake.loader, first));
    ASSERT_TRUE(seed_cache(root.url, cache_manifest('2'), fake.loader, second));
    ASSERT_TRUE(seed_cache(root.url, cache_manifest('3'), fake.loader, third));
    NSArray<NSString*>* ordered =
        [@[ first.key, second.key, third.key ] sortedArrayUsingSelector:@selector(compare:)];
    NSDictionary<NSString*, NSData*>* bookmarks =
        @{first.key : first.bookmark, second.key : second.bookmark, third.key : third.bookmark};
    NSMutableArray<NSData*>* evicted = [NSMutableArray array];
    BookmarkFakeLoader* loader = fake.loader;
    fake.loader.onEvict = ^(NSData* bookmark) {
      [evicted addObject:bookmark];
      loader.evictError = [bookmark isEqual:bookmarks[ordered[1]]]
                              ? [NSError errorWithDomain:NSPOSIXErrorDomain code:EBUSY userInfo:nil]
                              : nil;
    };
    StorageFaultScope fault(StorageOperation::BeforeEvict);
    EXPECT_EQ(clear_cache(root.url.path.UTF8String), Error::AccessFailed);
    drain_callbacks();
    EXPECT_EQ(fault.hits, 1);
    ASSERT_EQ(evicted.count, 2u);
    EXPECT_TRUE([evicted[0] isEqual:bookmarks[ordered[1]]]);
    EXPECT_TRUE([evicted[1] isEqual:bookmarks[ordered[2]]]);
    EXPECT_TRUE(cache_exists(test_bookmark_url(root.url, ordered[0])));
    EXPECT_TRUE(cache_exists(test_bookmark_url(root.url, ordered[1])));
    EXPECT_FALSE(cache_exists(test_bookmark_url(root.url, ordered[2])));
  }
}

}  // namespace
}  // namespace executorch::backends::coreai::testing
