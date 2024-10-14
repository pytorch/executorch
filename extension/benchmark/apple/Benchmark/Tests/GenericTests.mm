/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResourceTestCase.h"
#import "test_function.h"

#define ASSERT_OK_OR_RETURN(value__)            \
  ({                                            \
    XCTAssertEqual(value__.error(), Error::Ok); \
    if (!value__.ok()) {                        \
      return;                                   \
    }                                           \
  })

@interface GenericTests : ResourceTestCase
@end

@implementation GenericTests

+ (NSArray<NSString *> *)directories {
  return @[
    @"Resources",
    @"aatp/data", // AWS Farm devices look for resources here.
  ];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"model" : ^BOOL(NSString *filename){
      return [filename hasSuffix:@".mlpackage"];
    },
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:
    (NSDictionary<NSString *, NSString *> *)resources {
  NSString *modelPath = resources[@"model"];
  return @{
    @"load" : ^(XCTestCase *testCase){
      [testCase
          measureWithMetrics:@[ [XCTClockMetric new], [XCTMemoryMetric new] ]
                       block:^{
          XCTAssertEqual(prefill_no_kv_cache(), false);
                       }];
    }
  };
}

@end
