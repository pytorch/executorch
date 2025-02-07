/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "DynamicTestCase.h"

/**
 * ResourceTestCase is a subclass of DynamicTestCase that generates tests based
 * on resources. Subclasses should override the bundle, directories, predicates,
 * and dynamicTestsForResources methods.
 */
@interface ResourceTestCase : DynamicTestCase

/**
 * Returns an array of NSBundle objects to search for resources.
 * By default, returns the main bundle and the bundle for the class.
 */
+ (NSArray<NSBundle *> *)bundles;

/**
 * Returns an array of directory paths (relative to the bundles' resource paths)
 * to search. Subclasses should override to specify directories containing
 * resources.
 */
+ (NSArray<NSString *> *)directories;

/**
 * Returns a dictionary mapping resource keys to predicates.
 * Each predicate is a block that takes a filename and returns a BOOL indicating
 * a match. Subclasses should override to specify predicates for matching
 * resources.
 */
+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates;

/**
 * Returns a dictionary mapping test names to test blocks, given a dictionary of
 * resources. Subclasses should override to provide tests for combinations of
 * resources.
 *
 * @param resources A dictionary mapping resource keys to resource file paths.
 * @return A dictionary mapping test names to test blocks.
 */
+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:
    (NSDictionary<NSString *, NSString *> *)resources;

@end
