/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ETCoreAIBridge.h"

#import "CoreAIBridge-Swift.h"

BOOL ETCoreAIIsAvailable(void) {
  if (@available(macOS 27.0, iOS 27.0, *)) {
    return YES;
  }
  return NO;
}

NSString *ETCoreAIDeviceArchitectureName(void) {
  if (@available(macOS 27.0, iOS 27.0, *)) {
    return ETCoreAISwiftModelLoader.deviceArchitectureName;
  }
  return nil;
}

id<ETCoreAIModelLoading> ETCoreAICreateModelLoader(void) {
  return [[ETCoreAISwiftModelLoader alloc] init];
}
