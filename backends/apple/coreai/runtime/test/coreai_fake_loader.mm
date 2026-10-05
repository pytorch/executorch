/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_fake_loader.h"

using executorch::backends::coreai::testing::FakeBridgeState;

// ARC keeps failed asynchronous work from outliving its state storage.
@interface CoreAIFakeBridgeContext : NSObject {
 @public
  FakeBridgeState state;
}
@property(nonatomic, strong) dispatch_group_t callbacks;
@end

@implementation CoreAIFakeBridgeContext
- (instancetype)init {
  if ((self = [super init])) {
    self.callbacks = dispatch_group_create();
  }
  return self;
}
@end

namespace {
CoreAIFakeBridgeContext* current_context = nil;

NSError* fake_error(ETCoreAIErrorCode code) {
  return [NSError errorWithDomain:ETCoreAIErrorDomain code:code userInfo:nil];
}

// Release callback captures before publishing that all fake work has drained.
void run_async(CoreAIFakeBridgeContext* context, void (^work)()) {
  dispatch_group_t group = context.callbacks;
  __block void (^task)() = [work copy];
  dispatch_group_enter(group);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
    @autoreleasepool {
      task();
      task = nil;
    }
    dispatch_group_leave(group);
  });
}

::testing::AssertionResult wait_for_callbacks(CoreAIFakeBridgeContext* context,
                                              dispatch_time_t deadline) {
  if (dispatch_group_wait(context.callbacks, deadline) != 0) {
    return ::testing::AssertionFailure() << "Fake SDK callbacks did not finish";
  }
  return ::testing::AssertionSuccess();
}

dispatch_time_t cleanup_deadline() { return dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC); }
}  // namespace

@interface BookmarkFakeModel ()
@property(nonatomic, strong) CoreAIFakeBridgeContext* context;
- (instancetype)initWithContext:(CoreAIFakeBridgeContext*)context;
@end

@interface BookmarkFakeLoader ()
@property(nonatomic, strong) CoreAIFakeBridgeContext* context;
@end

@implementation BookmarkFakeModel
- (instancetype)init {
  return [self initWithContext:current_context];
}
- (instancetype)initWithContext:(CoreAIFakeBridgeContext*)context {
  if (context == nil) {
    return nil;
  }
  if ((self = [super init])) {
    self.context = context;
    ++context->state.prepared_models;
  }
  return self;
}
- (void)dealloc {
  if (_context != nil) {
    --_context->state.prepared_models;
  }
}
- (NSData*)copyBookmarkData {
  return self.owner.emptyBookmark ? [NSData data] : self.bookmark;
}
- (void)loadFunctionNamed:(NSString*)functionName
               inputNames:(NSArray<NSString*>*)inputNames
              outputNames:(NSArray<NSString*>*)outputNames
               completion:(void (^)(id<ETCoreAISession>, NSError*))completion {
  // Acquisition tests never bind functions.
  completion(nil, fake_error(ETCoreAIErrorUnsupported));
}
@end

@implementation BookmarkFakeLoader
- (instancetype)init {
  if (current_context == nil) {
    return nil;
  }
  if ((self = [super init])) {
    self.context = current_context;
    self.models = [NSMutableDictionary dictionary];
    restores = 0;
    specializations = 0;
    evictions = 0;
    ++_context->state.loaders;
  }
  return self;
}
- (void)dealloc {
  if (_context != nil) {
    --_context->state.loaders;
  }
}
- (void)restoreModelFromBookmark:(NSData*)bookmark
                      completion:(ETCoreAIAcquisitionCompletion)completion {
  ++restores;
  run_async(self.context, ^{
    @synchronized(self) {
      if (self.restoreError != nil) {
        completion(ETCoreAIAcquisitionStatusError, nil, self.restoreError);
        return;
      }
      BookmarkFakeModel* model = self.restoreMiss ? nil : self.models[bookmark];
      self.lastAcquired = model;
      completion(model == nil ? ETCoreAIAcquisitionStatusMiss : ETCoreAIAcquisitionStatusHit, model,
                 nil);
    }
  });
}
- (void)specializeModelAtURL:(NSURL*)url completion:(ETCoreAIAcquisitionCompletion)completion {
  ++specializations;
  run_async(self.context, ^{
    if (self.onSpecialize != nil) {
      self.onSpecialize(url);
    }
    @synchronized(self) {
      self.lastSpecializeURL = url.path;
      if (self.specializeError != nil) {
        completion(ETCoreAIAcquisitionStatusError, nil, self.specializeError);
        return;
      }
      if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) {
        ++self.context->state.missing_bundles;
        completion(ETCoreAIAcquisitionStatusError, nil, fake_error(ETCoreAIErrorInvalidModel));
        return;
      }
      BookmarkFakeModel* model = [[BookmarkFakeModel alloc] initWithContext:self.context];
      model.owner = self;
      model.bookmark = self.specializationBookmark != nil
                           ? self.specializationBookmark
                           : [NSUUID.UUID.UUIDString dataUsingEncoding:NSUTF8StringEncoding];
      self.models[model.bookmark] = model;
      self.lastAcquired = model;
      completion(ETCoreAIAcquisitionStatusHit, model, nil);
    }
  });
}
- (void)evictModelWithBookmark:(NSData*)bookmark completion:(void (^)(NSError*))completion {
  ++evictions;
  run_async(self.context, ^{
    if (self.onEvict != nil) {
      self.onEvict(bookmark);
    }
    @synchronized(self) {
      if (self.evictError == nil) {
        [self.models removeObjectForKey:bookmark];
      }
      completion(self.evictError);
    }
  });
}
- (void)clearRetainedState {
  self.onSpecialize = nil;
  self.onEvict = nil;
  self.lastAcquired = nil;
  [self.models removeAllObjects];
}
@end

namespace executorch::backends::coreai::testing {

ScopedFakeBridgeState::ScopedFakeBridgeState()
    : context_([[CoreAIFakeBridgeContext alloc] init]), previous_(current_context) {
  current_context = context_;
}

ScopedFakeBridgeState::~ScopedFakeBridgeState() {
  auto drained = wait_for_callbacks(cleanup_deadline());
  EXPECT_TRUE(drained);
  if (drained) {
    id<ETCoreAIModelLoading> loader = context_->state.bookmark_loader;
    if ([loader isKindOfClass:BookmarkFakeLoader.class]) {
      [(BookmarkFakeLoader*)loader clearRetainedState];
    }
    context_->state.bookmark_loader = nil;
    loader = nil;
    EXPECT_EQ(context_->state.prepared_models.load(), 0);
    EXPECT_EQ(context_->state.loaders.load(), 0);
    EXPECT_EQ(context_->state.missing_bundles.load(), 0);
  }
  EXPECT_EQ(current_context, context_);
  current_context = previous_;
}

FakeBridgeState& ScopedFakeBridgeState::state() { return context_->state; }

::testing::AssertionResult ScopedFakeBridgeState::wait_for_callbacks(dispatch_time_t deadline) {
  return ::wait_for_callbacks(context_, deadline);
}

BookmarkLoaderScope::BookmarkLoaderScope()
    : context_(current_context),
      previous_(context_ == nil ? nil : context_->state.bookmark_loader) {
  EXPECT_NE(context_, nil) << "BookmarkLoaderScope requires ScopedFakeBridgeState";
  if (context_ != nil) {
    loader = [[BookmarkFakeLoader alloc] init];
    context_->state.bookmark_loader = loader;
  }
}

BookmarkLoaderScope::~BookmarkLoaderScope() {
  if (context_ == nil) {
    return;
  }
  auto drained = ::wait_for_callbacks(context_, cleanup_deadline());
  EXPECT_TRUE(drained);
  if (drained) {
    [loader clearRetainedState];
  }
  EXPECT_EQ(context_->state.bookmark_loader, loader);
  context_->state.bookmark_loader = previous_;
  loader = nil;
}

}  // namespace executorch::backends::coreai::testing
