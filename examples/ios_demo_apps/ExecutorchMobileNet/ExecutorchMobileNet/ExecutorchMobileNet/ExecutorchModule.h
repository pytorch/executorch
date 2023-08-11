/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef ExecutorchModule_h
#define ExecutorchModule_h
#import <UIKit/UIKit.h>
#include <stdio.h>

#endif /* ExecutorchModule_h */

NS_ASSUME_NONNULL_BEGIN

@interface ExecutorchModule : NSObject
- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
- (char*)segmentImage:(void*)imageBuffer
            withWidth:(int)width
           withHeight:(int)height
    NS_SWIFT_NAME(segment(image:withWidth:withHeight:));
@end

NS_ASSUME_NONNULL_END
