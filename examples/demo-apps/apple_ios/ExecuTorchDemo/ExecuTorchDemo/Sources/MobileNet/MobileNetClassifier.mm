/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MobileNetClassifier.h"

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

NSErrorDomain const ETMobileNetClassifierErrorDomain =
    @"MobileNetClassifierErrorDomain";
const int32_t kSize = 224;
const int32_t kChannels = 3;

@implementation ETMobileNetClassifier {
  std::unique_ptr<Module> _module;
}

- (instancetype)initWithFilePath:(NSString*)filePath {
  self = [super init];
  if (self) {
    _module = std::make_unique<Module>(filePath.UTF8String);
  }
  return self;
}

- (BOOL)classifyWithInput:(float*)input
                   output:(float*)output
               outputSize:(NSInteger)outputSize
                    error:(NSError**)error {
  const auto result =
      _module->forward(from_blob(input, {1, kChannels, kSize, kSize}));

  if (!result.ok()) {
    if (error) {
      *error = [NSError
          errorWithDomain:ETMobileNetClassifierErrorDomain
                     code:NSInteger(result.error())
                 userInfo:@{
                   NSLocalizedDescriptionKey : [NSString
                       stringWithFormat:
                           @"Failed to run forward on the torch module, error code: %i",
                           result.error()]
                 }];
    }
    return NO;
  }
  const auto outputData = result->at(0).toTensor().const_data_ptr<float>();
  std::copy(outputData, outputData + outputSize, output);

  return YES;
}

@end
