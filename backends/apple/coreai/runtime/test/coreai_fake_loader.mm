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

bool valid_binding(CoreAIFakeBridgeContext* context, NSString* function, NSArray<NSString*>* inputs,
                   NSArray<NSString*>* outputs) {
  const bool valid = [function isEqualToString:@"main"] &&
                     [inputs isEqualToArray:@[ @"input_0" ]] &&
                     [outputs isEqualToArray:@[ @"output_0" ]];
  if (!valid) {
    ++context->state.binding_mismatches;
  }
  return valid;
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

@interface FakeSession ()
@property(nonatomic, strong) CoreAIFakeBridgeContext* context;
- (instancetype)initWithContext:(CoreAIFakeBridgeContext*)context;
@end

@implementation FakeSession
- (instancetype)init {
  return [self initWithContext:current_context];
}
- (instancetype)initWithContext:(CoreAIFakeBridgeContext*)context {
  if (context == nil) {
    return nil;
  }
  if ((self = [super init])) {
    self.context = context;
    ++context->state.sessions;
  }
  return self;
}
- (void)dealloc {
  if (_context != nil) {
    --_context->state.sessions;
  }
}
- (void)executeInputs:(NSArray<ETCoreAIInputTensor*>*)inputs
           completion:(void (^)(NSArray<ETCoreAITensor*>*, NSError*))completion {
  CoreAIFakeBridgeContext* context = self.context;
  ++context->state.executions;
  const bool should_fail = context->state.fail_execute;
  const bool malformed = context->state.bad_output;
  dispatch_semaphore_t ready = context->state.input_ready;
  dispatch_semaphore_t proceed = context->state.allow_input_read;
  run_async(context, ^{
    ETCoreAIInputTensor* input = inputs[0];
    context->state.last_input_bytes = input.bytes;
    context->state.last_input_byte_count = input.byteCount;
    if (ready != nil) {
      dispatch_semaphore_signal(ready);
      if (proceed == nil || dispatch_semaphore_wait(proceed, cleanup_deadline()) != 0) {
        ++context->state.input_wait_timeouts;
        completion(nil, fake_error(ETCoreAIErrorRuntime));
        return;
      }
    }
    if (input.byteCount > 0) {
      context->state.last_input_first_byte = static_cast<const unsigned char*>(input.bytes)[0];
    }
    if (should_fail) {
      completion(nil, fake_error(ETCoreAIErrorRuntime));
      return;
    }
    if (malformed) {
      completion(@[ [[ETCoreAITensor alloc] initWithData:[NSData data]
                                                   shape:@[ @2 ]
                                              scalarType:ETCoreAIScalarTypeFloat32] ],
                 nil);
      return;
    }
    NSData* data = [NSData dataWithBytes:input.bytes length:input.byteCount];
    ETCoreAITensor* output = [[ETCoreAITensor alloc] initWithData:data
                                                            shape:input.shape
                                                       scalarType:input.scalarType];
    completion(@[ output ], nil);
  });
}
@end

@interface FakePreparedModel ()
@property(nonatomic, strong) CoreAIFakeBridgeContext* context;
- (instancetype)initWithContext:(CoreAIFakeBridgeContext*)context;
@end

@implementation FakePreparedModel
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
  return self.bookmark;
}
- (void)loadFunctionNamed:(NSString*)functionName
               inputNames:(NSArray<NSString*>*)inputNames
              outputNames:(NSArray<NSString*>*)outputNames
               completion:(void (^)(id<ETCoreAISession>, NSError*))completion {
  CoreAIFakeBridgeContext* context = self.context;
  if (!valid_binding(context, functionName, inputNames, outputNames) || context->state.fail_load) {
    completion(nil, fake_error(ETCoreAIErrorInvalidModel));
    return;
  }
  FakeSession* session = [[FakeSession alloc] initWithContext:context];
  session.preparedPin = self;
  completion(session, nil);
}
@end

@interface FakeLoader ()
@property(nonatomic, strong) CoreAIFakeBridgeContext* context;
@end

@implementation FakeLoader
- (instancetype)init {
  if (current_context == nil) {
    return nil;
  }
  if ((self = [super init])) {
    self.context = current_context;
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
  FakePreparedModel* model = [[FakePreparedModel alloc] initWithContext:self.context];
  model.bookmark = bookmark;
  _context->state.last_bundle = [[NSString alloc] initWithData:bookmark
                                                      encoding:NSUTF8StringEncoding];
  completion(ETCoreAIAcquisitionStatusHit, model, nil);
}
- (void)specializeModelAtURL:(NSURL*)url completion:(ETCoreAIAcquisitionCompletion)completion {
  if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) {
    ++_context->state.missing_bundles;
    completion(ETCoreAIAcquisitionStatusError, nil, fake_error(ETCoreAIErrorInvalidModel));
    return;
  }
  _context->state.last_bundle = url.path;
  FakePreparedModel* model = [[FakePreparedModel alloc] initWithContext:self.context];
  model.bookmark = [url.path dataUsingEncoding:NSUTF8StringEncoding];
  completion(ETCoreAIAcquisitionStatusHit, model, nil);
}
- (void)evictModelWithBookmark:(NSData*)bookmark completion:(void (^)(NSError*))completion {
  completion(nil);
}
@end

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
  CoreAIFakeBridgeContext* context = self.context;
  BookmarkFakeLoader* owner = self.owner;
  if (!valid_binding(context, functionName, inputNames, outputNames) || owner == nil) {
    completion(nil, fake_error(ETCoreAIErrorInvalidModel));
    return;
  }
  ++owner->bindings;
  owner.lastBound = self;
  run_async(context, ^{
    if (owner.bindError != nil) {
      completion(nil, owner.bindError);
      return;
    }
    FakeSession* session = [[FakeSession alloc] initWithContext:context];
    session.preparedPin = self;
    completion(session, nil);
  });
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
    bindings = 0;
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
  self.lastBound = nil;
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
    context_->state.input_ready = nil;
    context_->state.allow_input_read = nil;
    EXPECT_EQ(context_->state.sessions.load(), 0);
    EXPECT_EQ(context_->state.prepared_models.load(), 0);
    EXPECT_EQ(context_->state.loaders.load(), 0);
    EXPECT_EQ(context_->state.binding_mismatches.load(), 0);
    EXPECT_EQ(context_->state.missing_bundles.load(), 0);
    EXPECT_EQ(context_->state.input_wait_timeouts.load(), 0);
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

BOOL ETCoreAIIsAvailable(void) {
  return current_context != nil && current_context->state.available;
}

NSString* ETCoreAIDeviceArchitectureName(void) {
  if (current_context == nil) {
    return nil;
  }
  ++current_context->state.architecture_queries;
  return current_context->state.device_architecture;
}

id<ETCoreAIModelLoading> ETCoreAICreateModelLoader(void) {
  if (current_context == nil || current_context->state.fail_loader_factory) {
    return nil;
  }
  return current_context->state.bookmark_loader ?: [[FakeLoader alloc] init];
}
