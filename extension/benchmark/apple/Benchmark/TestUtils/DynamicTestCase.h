/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <XCTest/XCTest.h>

/**
 * DynamicTestCase is a subclass of XCTestCase that allows dynamic creation of
 * test methods. Subclasses should override the `+dynamicTests` method to
 * provide a dictionary of test names and corresponding test blocks.
 */
@interface DynamicTestCase : XCTestCase

/**
 * Returns a dictionary mapping test names to test blocks.
 * Subclasses should override this method to provide dynamic tests.
 */
+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTests;

@end
