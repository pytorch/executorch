/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MobileNetClassifier.h"

#import <executorch/extension/runner/module.h>

using namespace ::torch::executor;

NSErrorDomain const ETMobileNetClassifierErrorDomain =
    @"MobileNetClassifierErrorDomain";
const int32_t kSize = 224;
const int32_t kChannels = 3;

@implementation ETMobileNetClassifier {
  std::unique_ptr<Module> _module;
}

- (nullable instancetype)initWithFilePath:(NSString*)filePath
                                    error:(NSError**)error {
  self = [super init];
  if (self) {
    try {
      _module = std::make_unique<Module>(filePath.UTF8String);
    } catch (const std::exception& exception) {
      if (error) {
        *error = [NSError
            errorWithDomain:ETMobileNetClassifierErrorDomain
                       code:-1
                   userInfo:@{
                     NSLocalizedDescriptionKey : [NSString
                         stringWithFormat:
                             @"Failed to initialize the torch module: %s",
                             exception.what()]
                   }];
      }
      return nil;
    }
  }
  return self;
}

- (BOOL)classifyWithInput:(float*)input
                   output:(float*)output
               outputSize:(NSInteger)outputSize
                    error:(NSError**)error {
  int32_t sizes[] = {1, kChannels, kSize, kSize};
  uint8_t order[] = {0, 1, 2, 3};
  int32_t strides[] = {kChannels * kSize * kSize, kSize * kSize, kSize, 1};
  TensorImpl tensorImpl(
      ScalarType::Float, std::size(sizes), sizes, input, order, strides);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto torchError = _module->forward(inputs, outputs);
  if (torchError != Error::Ok) {
    if (error) {
      *error = [NSError
          errorWithDomain:ETMobileNetClassifierErrorDomain
                     code:NSInteger(torchError)
                 userInfo:@{
                   NSLocalizedDescriptionKey : [NSString
                       stringWithFormat:
                           @"Failed to run forward on the torch module, error code: %i",
                           torchError]
                 }];
    }
    return NO;
  }
  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();
  std::copy(data, data + outputSize, output);

  return YES;
}

@end
