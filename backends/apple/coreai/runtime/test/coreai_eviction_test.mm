/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>
#include <pthread.h>
#include <cstdlib>
#include "coreai_bookmark_fixture.h"
#include "coreai_fake_loader.h"
#include "coreai_fault_scope.h"
#include "coreai_load_coordinator.h"
#include "coreai_source_fixture.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
NSError* bookmark_test_error(NSInteger code = EIO) {
  return [NSError errorWithDomain:NSPOSIXErrorDomain code:code userInfo:nil];
}

// These cases never retain a delegate handle beyond initialization.
Error initialize_and_destroy(NSDictionary* dict, const NamedDataMap* data, NSString* root) {
  if (root == nil) return Error::InvalidArgument;
  BackendOptions<1> options;
  auto error = options.set_option("coreai_assets_dir", root.UTF8String);
  if (error != Error::Ok) return error;
  auto* backend = get_backend_class("CoreAIBackend");
  if (backend == nullptr) return Error::Internal;
  BackendInitContext context(nullptr, nullptr, "forward", data,
                             {options.view().data(), options.view().size()});
  NSData* json = encode(dict);
  if (json == nil) return Error::InvalidArgument;
  FreeableBuffer processed(json.bytes, json.length, nullptr);
  auto handle = backend->init(context, &processed, {});
  if (!handle.ok()) return handle.error();
  backend->destroy(handle.get());
  return Error::Ok;
}

dispatch_time_t deadline() { return dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC); }

class CoreAIEvictionTest : public ::testing::Test {
 protected:
  ScopedFakeBridgeState bridge;
  void SetUp() override { runtime_init(); }
};

// Release the callback before waiting; never unwind storage borrowed by a hang.
struct ReleaseAndJoin final {
  dispatch_semaphore_t gate;
  dispatch_semaphore_t eviction_done;
  dispatch_semaphore_t loading_done;
  ScopedFakeBridgeState& bridge;
  pthread_t eviction{};
  pthread_t loading{};
  bool eviction_started = false;
  bool loading_started = false;
  bool released = false;
  bool joined = false;
  long eviction_wait = -1;
  long loading_wait = -1;

  void release() {
    if (!released) {
      dispatch_semaphore_signal(gate);
      released = true;
    }
  }
  void join() {
    release();
    if (joined) return;
    const auto completion_deadline = deadline();
    if (eviction_started)
      eviction_wait = dispatch_semaphore_wait(eviction_done, completion_deadline);
    if (loading_started) loading_wait = dispatch_semaphore_wait(loading_done, completion_deadline);
    if ((eviction_started && eviction_wait != 0) || (loading_started && loading_wait != 0)) {
      ADD_FAILURE() << "Eviction/load workers did not return after releasing the gate";
      std::_Exit(EXIT_FAILURE);
    }
    const int eviction_join = eviction_started ? pthread_join(eviction, nullptr) : 0;
    const int loading_join = loading_started ? pthread_join(loading, nullptr) : 0;
    if (eviction_join != 0 || loading_join != 0) {
      ADD_FAILURE() << "Cannot join eviction/load workers: " << eviction_join << ", "
                    << loading_join;
      std::_Exit(EXIT_FAILURE);
    }
    auto drained = bridge.wait_for_callbacks(deadline());
    if (!drained) {
      ADD_FAILURE() << drained.message();
      std::_Exit(EXIT_FAILURE);
    }
    joined = true;
  }
  ~ReleaseAndJoin() { join(); }
};
}  // namespace

@interface PinBookmarkLoader : BookmarkFakeLoader
@property(nonatomic, weak) BookmarkFakeModel* temporary;
@property(nonatomic, copy) ETCoreAIAcquisitionCompletion retainedCompletion;
@end

@implementation PinBookmarkLoader
- (void)specializeModelAtURL:(NSURL*)url completion:(ETCoreAIAcquisitionCompletion)completion {
  ++specializations;
  self.retainedCompletion = completion;
  @autoreleasepool {
    BookmarkFakeModel* model = [[BookmarkFakeModel alloc] init];
    model.owner = self;
    model.bookmark = [@"pinned-bookmark" dataUsingEncoding:NSUTF8StringEncoding];
    self.temporary = model;
    completion(ETCoreAIAcquisitionStatusHit, model, nil);
    model = nil;
  }
}
- (void)restoreModelFromBookmark:(NSData*)bookmark
                      completion:(ETCoreAIAcquisitionCompletion)completion {
  ++restores;
  self.retainedCompletion = completion;
  @autoreleasepool {
    BookmarkFakeModel* model = self.temporary;
    completion(model == nil ? ETCoreAIAcquisitionStatusMiss : ETCoreAIAcquisitionStatusHit, model,
               nil);
    model = nil;
  }
}
- (void)evictModelWithBookmark:(NSData*)bookmark completion:(void (^)(NSError*))completion {
  ++evictions;
  bool busy;
  @autoreleasepool {
    busy = self.temporary != nil;
  }
  completion(busy ? bookmark_test_error(EBUSY) : nil);
}
- (void)clearRetainedState {
  self.retainedCompletion = nil;
  [super clearRetainedState];
}
@end

namespace {
TEST_F(CoreAIEvictionTest, ConfirmedMissRecoveryEvictsOnlyCurrentBookmark) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    BookmarkLoaderScope fake;
    ASSERT_NE(fake.loader, nil);
    TestData data;
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    auto key_result = test_bookmark_key(selected.get());
    ASSERT_TRUE(key_result.ok());
    NSString* key = key_result.get();
    NSMutableDictionary* dict = manifest_dict();
    ASSERT_EQ(initialize_and_destroy(dict, &data, root.url.path), Error::Ok);
    auto original_result = saved_bookmark(root.url, key);
    ASSERT_TRUE(original_result.ok());
    NSData* original = original_result.get();
    ASSERT_NE(original, nil);
    NSString* source_path = fake.loader.lastSpecializeURL;
    ASSERT_NE(source_path, nil);
    NSURL* source_url = [NSURL fileURLWithPath:source_path];
    NSDictionary* sources = nil;
    ASSERT_TRUE(asset_tree_snapshot(source_url, sources));

    fake.loader.restoreMiss = YES;
    fake.loader.specializeError = bookmark_test_error();
    EXPECT_NE(initialize_and_destroy(dict, &data, root.url.path), Error::Ok);
    auto after_failure = saved_bookmark(root.url, key);
    ASSERT_TRUE(after_failure.ok());
    EXPECT_TRUE([after_failure.get() isEqual:original]);
    EXPECT_TRUE(snapshot_matches(source_url, sources));
    fake.loader.specializeError = nil;
    ASSERT_EQ(initialize_and_destroy(dict, &data, root.url.path), Error::Ok);
    EXPECT_EQ(fake.loader->specializations.load(), 3);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    EXPECT_TRUE([fake.loader.lastSpecializeURL isEqual:source_path]);
    auto saved_result = saved_bookmark(root.url, key);
    ASSERT_TRUE(saved_result.ok());
    NSData* saved = saved_result.get();
    ASSERT_NE(saved, nil);
    EXPECT_FALSE([saved isEqual:original]);
    EXPECT_TRUE(snapshot_matches(source_url, sources));
    ASSERT_EQ(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
    EXPECT_NE(fake.loader.models[original], nil);
    EXPECT_EQ(fake.loader.models[saved], nil);
    EXPECT_TRUE(snapshot_matches(source_url, sources));
    fake.loader.restoreMiss = NO;
    fake.loader.emptyBookmark = YES;
    EXPECT_NE(initialize_and_destroy(dict, &data, root.url.path), Error::Ok);
    auto empty = saved_bookmark(root.url, key);
    ASSERT_TRUE(empty.ok());
    EXPECT_EQ(empty.get(), nil);
    fake.loader.emptyBookmark = NO;
    ASSERT_EQ(initialize_and_destroy(dict, &data, root.url.path), Error::Ok);
    EXPECT_EQ(fake.loader->specializations.load(), 5);
    EXPECT_EQ(data.requests.load(), data.releases.load());
  }
}

TEST_F(CoreAIEvictionTest, BusyAndUnknownEvictionRetainBookmarkUntilConfirmedAbsent) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    BookmarkLoaderScope fake;
    ASSERT_NE(fake.loader, nil);
    TestData data;
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    auto key_result = test_bookmark_key(selected.get());
    ASSERT_TRUE(key_result.ok());
    NSString* key = key_result.get();
    EXPECT_EQ(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    ASSERT_EQ(initialize_and_destroy(manifest_dict(), &data, root.url.path), Error::Ok);
    auto saved_result = saved_bookmark(root.url, key);
    ASSERT_TRUE(saved_result.ok());
    NSData* saved = saved_result.get();
    ASSERT_NE(saved, nil);
    NSURL* source_url =
        [[[root.url URLByAppendingPathComponent:@"staging"] URLByAppendingPathComponent:key]
            URLByAppendingPathComponent:selected->path.lastPathComponent];
    NSDictionary* source = nil;
    ASSERT_TRUE(asset_tree_snapshot(source_url, source));
    for (bool probe_error : {false, true}) {
      SCOPED_TRACE(probe_error ? "unknown restore failure" : "restore hit/busy");
      fake.loader.evictError = bookmark_test_error(EBUSY);
      fake.loader.restoreError = probe_error ? bookmark_test_error() : nil;
      EXPECT_NE(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
      auto retained = saved_bookmark(root.url, key);
      ASSERT_TRUE(retained.ok());
      EXPECT_TRUE([retained.get() isEqual:saved]);
      EXPECT_TRUE(snapshot_matches(source_url, source));
    }
    fake.loader.restoreError = nil;
    fake.loader.evictError = nil;
    {
      StorageFaultScope fault(StorageOperation::AfterEvict);
      EXPECT_NE(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
      EXPECT_EQ(fault.hits, 1);
    }
    auto retained = saved_bookmark(root.url, key);
    ASSERT_TRUE(retained.ok());
    EXPECT_TRUE([retained.get() isEqual:saved]);
    fake.loader.evictError = bookmark_test_error(ENOENT);
    ASSERT_EQ(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
    auto absent = saved_bookmark(root.url, key);
    ASSERT_TRUE(absent.ok());
    EXPECT_EQ(absent.get(), nil);
    EXPECT_TRUE(snapshot_matches(source_url, source));
    fake.loader.evictError = nil;
    ASSERT_EQ(initialize_and_destroy(manifest_dict(), &data, root.url.path), Error::Ok);
    {
      StorageFaultScope fault(StorageOperation::Remove);
      EXPECT_NE(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
      EXPECT_EQ(fault.hits, 1);
      auto before_remove = saved_bookmark(root.url, key);
      ASSERT_TRUE(before_remove.ok());
      EXPECT_NE(before_remove.get(), nil);
    }
    ASSERT_EQ(evict_bookmark_model(root.url.path, key, fake.loader), Error::Ok);
    auto removed = saved_bookmark(root.url, key);
    ASSERT_TRUE(removed.ok());
    EXPECT_EQ(removed.get(), nil);
    EXPECT_TRUE(snapshot_matches(source_url, source));
    EXPECT_EQ(data.requests.load(), data.releases.load());
  }
}

TEST_F(CoreAIEvictionTest, PreparedPinsBlockEvictionWithoutRetainedCompletionLeak) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    TestData data;
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    auto key_result = test_bookmark_key(selected.get());
    ASSERT_TRUE(key_result.ok());
    NSString* key = key_result.get();
    PinBookmarkLoader* loader = [[PinBookmarkLoader alloc] init];
    ASSERT_NE(loader, nil);
    // Scope teardown clears retained completions even after fatal test
    // failures.
    bridge.state().bookmark_loader = loader;
    @autoreleasepool {
      auto pinned =
          acquire_bookmark_model(selected.get(), &data, root.url.path, @"macOS", @"arch_b", loader);
      ASSERT_TRUE(pinned.ok());
      EXPECT_NE(loader.temporary, nil);
      EXPECT_NE(evict_bookmark_model(root.url.path, key, loader), Error::Ok);
      auto saved = saved_bookmark(root.url, key);
      ASSERT_TRUE(saved.ok());
      EXPECT_NE(saved.get(), nil);
    }
    EXPECT_EQ(loader.temporary, nil);
    EXPECT_NE(loader.retainedCompletion, nil);
    ASSERT_EQ(evict_bookmark_model(root.url.path, key, loader), Error::Ok);
    auto removed = saved_bookmark(root.url, key);
    ASSERT_TRUE(removed.ok());
    EXPECT_EQ(removed.get(), nil);
    EXPECT_EQ(loader->evictions.load(), 2);
    NSURL* source_url =
        [[[root.url URLByAppendingPathComponent:@"staging"] URLByAppendingPathComponent:key]
            URLByAppendingPathComponent:selected->path.lastPathComponent];
    EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:source_url.path]);
    [loader clearRetainedState];
    EXPECT_EQ(data.requests.load(), data.releases.load());
  }
}

TEST_F(CoreAIEvictionTest, LoadContendsOnActualKeyLockUntilEvictionFinishes) {
  @autoreleasepool {
    BookmarkDirectory root;
    ASSERT_NE(root.url, nil);
    BookmarkLoaderScope fake;
    ASSERT_NE(fake.loader, nil);
    TestData data;
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    auto key_result = test_bookmark_key(selected.get());
    ASSERT_TRUE(key_result.ok());
    NSString* key = key_result.get();
    ASSERT_EQ(initialize_and_destroy(manifest_dict(), &data, root.url.path), Error::Ok);
    dispatch_semaphore_t deleting = dispatch_semaphore_create(0);
    dispatch_semaphore_t proceed = dispatch_semaphore_create(0);
    dispatch_semaphore_t loading_observed = dispatch_semaphore_create(0);
    dispatch_semaphore_t eviction_done = dispatch_semaphore_create(0);
    dispatch_semaphore_t loading_done = dispatch_semaphore_create(0);
    __block long gate_wait = -1;
    fake.loader.onEvict = ^(NSData*) {
      dispatch_semaphore_signal(deleting);
      gate_wait = dispatch_semaphore_wait(proceed, deadline());
    };
    Error eviction_error = Error::Internal;
    bool load_ok = false;
    auto evict = [&] {
      @autoreleasepool {
        eviction_error = evict_bookmark_model(root.url.path, key, fake.loader);
      }
      dispatch_semaphore_signal(eviction_done);
    };
    auto load = [&] {
      @autoreleasepool {
        auto acquired = acquire_bookmark_model(selected.get(), &data, root.url.path, @"macOS",
                                               @"arch_b", fake.loader);
        load_ok = acquired.ok() && acquired.get() != nil;
      }
      dispatch_semaphore_signal(loading_observed);
      dispatch_semaphore_signal(loading_done);
    };
    ReleaseAndJoin cleanup{proceed, eviction_done, loading_done, bridge};
    const int eviction_start = pthread_create(
        &cleanup.eviction, nullptr,
        [](void* context) -> void* {
          (*static_cast<decltype(evict)*>(context))();
          return nullptr;
        },
        &evict);
    cleanup.eviction_started = eviction_start == 0;
    const long entered =
        cleanup.eviction_started ? dispatch_semaphore_wait(deleting, deadline()) : -1;
    int loading_start = -1;
    long premature_completion = -1;
    if (entered == 0) {
      loading_start = pthread_create(
          &cleanup.loading, nullptr,
          [](void* context) -> void* {
            (*static_cast<decltype(load)*>(context))();
            return nullptr;
          },
          &load);
      cleanup.loading_started = loading_start == 0;
      if (cleanup.loading_started) {
        premature_completion = dispatch_semaphore_wait(
            loading_observed, dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC));
      }
    }
    cleanup.join();
    fake.loader.onEvict = nil;

    EXPECT_EQ(eviction_start, 0);
    EXPECT_EQ(loading_start, 0);
    EXPECT_EQ(entered, 0);
    EXPECT_NE(premature_completion, 0);
    EXPECT_EQ(gate_wait, 0);
    EXPECT_EQ(cleanup.eviction_wait, 0);
    EXPECT_EQ(cleanup.loading_wait, 0);
    EXPECT_EQ(eviction_error, Error::Ok);
    EXPECT_TRUE(load_ok);
    EXPECT_EQ(fake.loader->specializations.load(), 2);
    auto saved = saved_bookmark(root.url, key);
    ASSERT_TRUE(saved.ok());
    EXPECT_NE(saved.get(), nil);
    EXPECT_EQ(data.requests.load(), data.releases.load());
  }
}
}  // namespace
