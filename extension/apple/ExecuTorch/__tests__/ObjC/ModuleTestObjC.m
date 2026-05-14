/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <XCTest/XCTest.h>

#import <ExecuTorch/ExecuTorch.h>

@interface ModuleTestObjC : XCTestCase
@end

@implementation ModuleTestObjC

- (NSBundle *)resourceBundle {
#if SWIFT_PACKAGE
  return SWIFTPM_MODULE_BUNDLE;
#else
  return [NSBundle bundleForClass:[self class]];
#endif
}

// Resolves a fixture by name. In CI (the `CI` env var is set, regardless
// of value — matches the convention used by GitHub Actions / Sandcastle /
// most CI systems), absence is a hard failure (`XCTFail`). Locally, absence
// is a soft skip — convenient on dev machines without the CoreML python
// deps. Returns nil when the test should not proceed; the caller should
// `return` immediately on nil.
- (nullable NSString *)requireFixture:(NSString *)name ofType:(NSString *)type {
  NSString *path = [[self resourceBundle] pathForResource:name ofType:type];
  if (path) return path;
  NSString *message = [NSString stringWithFormat:@"%@.%@ not bundled.", name, type];
  if (NSProcessInfo.processInfo.environment[@"CI"] != nil) {
    // CI: hard fail. XCTFail records the failure; we then return nil so the
    // caller exits early. Single failure artifact.
    XCTFail(@"[CI] %@", message);
  } else {
    // Local: skip. XCTSkip throws an exception that XCTest catches and
    // marks the test as skipped — the `return nil` below is unreachable.
    XCTSkip(@"%@", message);
  }
  return nil;
}

// Mirrors the Swift testLoadWithBackendOptionsThenExecuteOnCoreMLDelegatedModel,
// exercising the ObjC API surface directly so coverage does not depend on
// the Swift overlays.
- (void)testLoadWithOptionsThenExecuteOnCoreMLDelegatedModel {
  NSString *modelPath = [self requireFixture:@"add_coreml" ofType:@"pte"];
  if (!modelPath) return;
  NSError *error = nil;
  ExecuTorchBackendOptionsMap *options = [ExecuTorchBackendOptionsMap mapWithOptions:@{
    @"CoreMLBackend": @[
      [ExecuTorchBackendOption optionWithKey:@"compute_unit" stringValue:@"cpu_and_gpu"],
      [ExecuTorchBackendOption optionWithKey:@"_use_new_cache" booleanValue:YES],
    ]
  } error:&error];
  XCTAssertNotNil(options, @"%@", error);

  ExecuTorchModule *module = [[ExecuTorchModule alloc] initWithFilePath:modelPath];
  XCTAssertTrue([module loadWithOptions:options error:&error], @"%@", error);

  // No explicit -loadMethod: — exercise the lazy load_method path that
  // consumes the retained backend options map.
  ExecuTorchTensor *one =
      [[ExecuTorchTensor alloc] initWithScalars:@[@1.0f] dataType:ExecuTorchDataTypeFloat];
  NSArray<ExecuTorchValue *> *outputs =
      [module forwardWithTensors:@[one, one] error:&error];
  XCTAssertNotNil(outputs, @"%@", error);
  XCTAssertEqual(outputs.count, 1u);

  __block float result = NAN;
  [outputs.firstObject.tensorValue
      bytesWithHandler:^(const void *bytes, NSInteger count, ExecuTorchDataType dt) {
    if (dt == ExecuTorchDataTypeFloat && count >= 1) {
      result = ((const float *)bytes)[0];
    }
  }];
  XCTAssertEqual(result, 2.0f);
}

// Validation: oversized integer / oversized key / oversized string value
// must each surface as nil + populated NSError, never silently truncate.
- (void)testBackendOptionsMapValidation {
  NSError *error = nil;
  // Oversized integer.
  // NOTE: dict/array literals are extracted to locals because the C
  // preprocessor only treats `()` as nesting — commas inside `@{...}` and
  // `@[...]` would otherwise split the XCTAssertNil(...) macro argument.
  long long oversized = (long long)INT32_MAX + 1;
  ExecuTorchBackendOptionsMap *oversizedIntMap =
      [ExecuTorchBackendOptionsMap mapWithOptions:@{
        @"AnyBackend": @[
          [ExecuTorchBackendOption optionWithKey:@"too_big" integerValue:(NSInteger)oversized],
        ]
      } error:&error];
  XCTAssertNil(oversizedIntMap);
  XCTAssertNotNil(error);

  // Oversized key.
  error = nil;
  NSString *longKey = [@"" stringByPaddingToLength:256 withString:@"k" startingAtIndex:0];
  ExecuTorchBackendOptionsMap *oversizedKeyMap =
      [ExecuTorchBackendOptionsMap mapWithOptions:@{
        @"AnyBackend": @[
          [ExecuTorchBackendOption optionWithKey:longKey integerValue:1],
        ]
      } error:&error];
  XCTAssertNil(oversizedKeyMap);
  XCTAssertNotNil(error);

  // Oversized string value.
  error = nil;
  NSString *longValue = [@"" stringByPaddingToLength:4096 withString:@"v" startingAtIndex:0];
  ExecuTorchBackendOptionsMap *oversizedValueMap =
      [ExecuTorchBackendOptionsMap mapWithOptions:@{
        @"AnyBackend": @[
          [ExecuTorchBackendOption optionWithKey:@"compute_unit" stringValue:longValue],
        ]
      } error:&error];
  XCTAssertNil(oversizedValueMap);
  XCTAssertNotNil(error);
}

// A single map can be reused across multiple Module instances. Each Module
// retains the options independently via ARC.
- (void)testBackendOptionsMapReusedAcrossModules {
  NSString *modelPath = [self requireFixture:@"add_coreml" ofType:@"pte"];
  if (!modelPath) return;
  NSError *error = nil;
  ExecuTorchBackendOptionsMap *options = [ExecuTorchBackendOptionsMap mapWithOptions:@{
    @"CoreMLBackend": @[
      [ExecuTorchBackendOption optionWithKey:@"compute_unit" stringValue:@"cpu_only"],
    ]
  } error:&error];
  XCTAssertNotNil(options, @"%@", error);

  ExecuTorchTensor *one =
      [[ExecuTorchTensor alloc] initWithScalars:@[@1.0f] dataType:ExecuTorchDataTypeFloat];

  for (NSInteger i = 0; i < 2; ++i) {
    ExecuTorchModule *module = [[ExecuTorchModule alloc] initWithFilePath:modelPath];
    XCTAssertTrue([module loadWithOptions:options error:&error], @"%@", error);
    NSArray<ExecuTorchValue *> *outputs =
        [module forwardWithTensors:@[one, one] error:&error];
    XCTAssertNotNil(outputs, @"%@", error);
    __block float result = NAN;
    [outputs.firstObject.tensorValue
        bytesWithHandler:^(const void *bytes, NSInteger count, ExecuTorchDataType dt) {
      if (dt == ExecuTorchDataTypeFloat && count >= 1) {
        result = ((const float *)bytes)[0];
      }
    }];
    XCTAssertEqual(result, 2.0f);
  }
}

// Covers -[ExecuTorchModule loadMethod:options:error:] and locks the
// "load_method consumes the borrow synchronously and does not cache it"
// invariant that the no-retain decision in the wrapper rests on.
// See the citation on ExecuTorchModule.mm's loadMethod:options:error:
// (module.cpp#L353-L409). If a future refactor caches `backend_options`
// on the Module, this test fails (the weak reference stays non-nil).
- (void)testLoadMethodWithOptionsDoesNotRetainOptions {
  NSString *modelPath = [self requireFixture:@"add_coreml" ofType:@"pte"];
  if (!modelPath) return;
  NSError *error = nil;
  ExecuTorchModule *module = [[ExecuTorchModule alloc] initWithFilePath:modelPath];

  __weak ExecuTorchBackendOptionsMap *weakOptions = nil;
  @autoreleasepool {
    ExecuTorchBackendOptionsMap *options = [ExecuTorchBackendOptionsMap mapWithOptions:@{
      @"CoreMLBackend": @[
        [ExecuTorchBackendOption optionWithKey:@"compute_unit" stringValue:@"cpu_and_gpu"],
      ],
    } error:&error];
    XCTAssertNotNil(options, @"%@", error);
    weakOptions = options;
    XCTAssertTrue([module loadMethod:@"forward" options:options error:&error],
                  @"%@", error);
    XCTAssertTrue([module isMethodLoaded:@"forward"]);
  }
  // The local + any autoreleased refs have drained. If loadMethod:options:
  // silently retained the map, weakOptions would still be live here.
  XCTAssertNil(weakOptions,
      @"loadMethod:options: must not retain the map (load_method consumes "
      @"it synchronously). See module.cpp load_method borrow contract.");

  // The loaded method must still run — it reads from _module->methods_
  // (populated during loadMethod:options:), not from the options map.
  ExecuTorchTensor *one =
      [[ExecuTorchTensor alloc] initWithScalars:@[@1.0f] dataType:ExecuTorchDataTypeFloat];
  NSArray<ExecuTorchValue *> *outputs =
      [module forwardWithTensors:@[one, one] error:&error];
  XCTAssertNotNil(outputs, @"%@", error);

  __block float result = NAN;
  [outputs.firstObject.tensorValue
      bytesWithHandler:^(const void *bytes, NSInteger count, ExecuTorchDataType dt) {
    if (dt == ExecuTorchDataTypeFloat && count >= 1) {
      result = ((const float *)bytes)[0];
    }
  }];
  XCTAssertEqual(result, 2.0f);
}

@end
