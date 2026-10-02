/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>
#include <atomic>
#import "ETCoreAIBridge.h"

@class CoreAIFakeBridgeContext;
@class BookmarkFakeLoader;

namespace executorch::backends::coreai::testing {

// Configure on the test thread before starting workers; join before changing
// it.
struct FakeBridgeState {
  id<ETCoreAIModelLoading> bookmark_loader = nil;
  std::atomic<int> prepared_models{0};
  std::atomic<int> loaders{0};
  // Worker-side contract failures are checked by the scope on the test thread.
  std::atomic<int> missing_bundles{0};
};

// Process-visible, not thread-local. Scopes and configuration changes are
// serial.
class ScopedFakeBridgeState final {
 public:
  ScopedFakeBridgeState();
  ~ScopedFakeBridgeState();
  FakeBridgeState& state();
  // Join caller threads before draining callbacks.
  ::testing::AssertionResult wait_for_callbacks(dispatch_time_t deadline);
  ScopedFakeBridgeState(const ScopedFakeBridgeState&) = delete;
  ScopedFakeBridgeState& operator=(const ScopedFakeBridgeState&) = delete;

 private:
  CoreAIFakeBridgeContext* context_;
  CoreAIFakeBridgeContext* previous_;
};

}  // namespace executorch::backends::coreai::testing

@interface BookmarkFakeModel : NSObject <ETCoreAIPreparedModel>
@property(nonatomic, weak) BookmarkFakeLoader* owner;
@property(nonatomic, copy) NSData* bookmark;
@end

@interface BookmarkFakeLoader : NSObject <ETCoreAIModelLoading> {
 @public
  std::atomic<int> restores;
  std::atomic<int> specializations;
  std::atomic<int> evictions;
}
@property(nonatomic, strong) NSMutableDictionary<NSData*, BookmarkFakeModel*>* models;
@property(nonatomic, strong) NSError* restoreError;
@property(nonatomic, strong) NSError* specializeError;
@property(nonatomic, strong) NSError* evictError;
@property(nonatomic) BOOL restoreMiss;
@property(nonatomic) BOOL emptyBookmark;
@property(nonatomic, copy) NSData* specializationBookmark;
@property(nonatomic, copy) void (^onSpecialize)(NSURL*);
@property(nonatomic, copy) void (^onEvict)(NSData*);
@property(nonatomic, strong) BookmarkFakeModel* lastAcquired;
@property(nonatomic, copy) NSString* lastSpecializeURL;
// Call only after workers/callbacks finish and caller-owned handles are
// released.
- (void)clearRetainedState;
@end

namespace executorch::backends::coreai::testing {

struct BookmarkLoaderScope final {
  BookmarkFakeLoader* loader = nil;
  BookmarkLoaderScope();
  ~BookmarkLoaderScope();
  BookmarkLoaderScope(const BookmarkLoaderScope&) = delete;
  BookmarkLoaderScope& operator=(const BookmarkLoaderScope&) = delete;

 private:
  CoreAIFakeBridgeContext* context_;
  id<ETCoreAIModelLoading> previous_;
};

}  // namespace executorch::backends::coreai::testing
