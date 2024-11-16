/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResourceTestCase.h"

static void generateCombinations(
    NSDictionary<NSString *, NSArray<NSString *> *> *matchesByResource,
    NSArray<NSString *> *keys,
    NSMutableDictionary<NSString *, NSString *> *result,
    NSUInteger index,
    NSMutableSet<NSDictionary<NSString *, NSString *> *> *combinations) {
  if (index == keys.count) {
    if (result.count == keys.count) {
      [combinations addObject:[result copy]];
    }
    return;
  }
  NSString *key = keys[index];
  NSArray<NSString *> *matches = matchesByResource[key] ?: @[];
  if (!matches.count) {
    generateCombinations(
        matchesByResource, keys, result, index + 1, combinations);
    return;
  }
  for (NSString *match in matches) {
    result[key] = match;
    generateCombinations(
        matchesByResource, keys, result, index + 1, combinations);
    [result removeObjectForKey:key];
  }
}

@implementation ResourceTestCase

+ (NSArray<NSBundle *> *)bundles {
  return @[ [NSBundle mainBundle], [NSBundle bundleForClass:self] ];
}

+ (NSArray<NSString *> *)directories {
  return @[];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{};
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:
    (NSDictionary<NSString *, NSString *> *)resources {
  return @{};
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTests {
  NSMutableDictionary<NSString *, void (^)(XCTestCase *)> *tests =
      [NSMutableDictionary new];
  NSMutableSet<NSDictionary<NSString *, NSString *> *> *combinations =
      [NSMutableSet new];
  NSDictionary<NSString *, BOOL (^)(NSString *)> *predicates =
      [self predicates];
  NSArray<NSString *> *sortedKeys =
      [predicates.allKeys sortedArrayUsingSelector:@selector(compare:)];

  if (predicates.count == 0)
    return @{};

  for (NSBundle *bundle in self.bundles) {
    for (NSString *directory in self.directories) {
      NSArray<NSURL *> *resourceURLs =
          [bundle URLsForResourcesWithExtension:nil subdirectory:directory];
      if (!resourceURLs.count) {
        continue;
      };
      NSMutableDictionary<NSString *, NSMutableArray<NSString *> *>
          *matchesByResource = [NSMutableDictionary new];

      for (NSURL *url in resourceURLs) {
        NSString *file = url.lastPathComponent;
        NSString *fullPath = url.path;

        for (NSString *key in sortedKeys) {
          if (predicates[key](file)) {
            matchesByResource[key] =
                matchesByResource[key] ?: [NSMutableArray new];
            [matchesByResource[key] addObject:fullPath];
          }
        }
      }
      NSMutableDictionary<NSString *, NSString *> *result =
          [NSMutableDictionary new];
      generateCombinations(
          matchesByResource, sortedKeys, result, 0, combinations);
    }
  }
  for (NSDictionary<NSString *, NSString *> *resources in combinations) {
    NSMutableString *resourceString = [NSMutableString new];
    NSCharacterSet *punctuationSet = [NSCharacterSet punctuationCharacterSet];
    for (NSString *key in sortedKeys) {
      NSString *lastComponent = [resources[key] lastPathComponent];
      NSString *cleanedComponent =
          [[lastComponent componentsSeparatedByCharactersInSet:punctuationSet]
              componentsJoinedByString:@"_"];
      [resourceString appendFormat:@"_%@", cleanedComponent];
    }
    NSDictionary<NSString *, void (^)(XCTestCase *)> *resourceTests =
        [self dynamicTestsForResources:resources];
    [resourceTests
        enumerateKeysAndObjectsUsingBlock:^(
            NSString *testName, void (^testBlock)(XCTestCase *), BOOL *stop) {
          tests[[testName stringByAppendingString:resourceString]] = testBlock;
        }];
  }
  return tests;
}

@end
