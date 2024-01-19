/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ETMobileNetClassifierErrorDomain;

@interface ETMobileNetClassifier : NSObject

- (instancetype)initWithFilePath:(NSString*)filePath;
- (BOOL)classifyWithInput:(float*)input
                   output:(float*)output
               outputSize:(NSInteger)predictionBufferSize
                    error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
