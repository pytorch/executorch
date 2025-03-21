/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecutorchRuntimeTensorValue.h"

#import <memory>

#import <executorch/extension/module/module.h>

using torch::executor::TensorImpl;
using torch::executor::ScalarType;

@implementation ExecutorchRuntimeTensorValue
{
  std::unique_ptr<TensorImpl> _tensor;
  // TensorImpl DOES NOT take ownership.
  // This float vector is what keeps the data in memory.
  std::vector<float> _floatData;
  std::vector<int32_t> _shape;
}

- (instancetype)initWithData:(std::vector<float>)floatData
                       shape:(std::vector<int32_t>)shape
{
  if (self = [super init]) {
    _floatData.assign(floatData.begin(), floatData.end());
    _shape.assign(shape.begin(), shape.end());
    _tensor = std::make_unique<TensorImpl>(ScalarType::Float, std::size(_shape), _shape.data(), _floatData.data());
  }
  return self;
}

- (instancetype)initWithFloatArray:(NSArray<NSNumber *> *)floatArray shape:(NSArray<NSNumber *> *)shape
{
  std::vector<float> floatVector;
  std::vector<int32_t> shapeVector;

  floatVector.reserve(floatArray.count);
  for (int i = 0; i < floatArray.count; i++) {
    floatVector.push_back([floatArray[i] floatValue]);
  }
  shapeVector.reserve(shape.count);
  for (int i = 0; i < shape.count; i++) {
    shapeVector.push_back([shape[i] intValue]);
  }

  return [self initWithData:floatVector shape:shapeVector];
}

- (nullable instancetype)initWithTensor:(torch::executor::Tensor)tensor error:(NSError * _Nullable * _Nullable)error
{
  if (tensor.scalar_type() != ScalarType::Float) {
    if (error) {
      *error = [ModelRuntimeValueErrorFactory invalidType:[NSString stringWithFormat:@"torch::executor::ScalarType::%hhd", tensor.scalar_type()] expectedType:@"torch::executor::ScalarType::Float"];
    }
    return nil;
  }

  std::vector<float> floatVector;
  std::vector<int32_t> shapeVector;
  shapeVector.assign(tensor.sizes().begin(), tensor.sizes().end());
  floatVector.assign(tensor.const_data_ptr<float>(), tensor.const_data_ptr<float>() + tensor.numel());
  return [self initWithData:floatVector shape:shapeVector];
}

- (nullable ModelRuntimeTensorValueBridgingTuple *)floatRepresentationAndReturnError:(NSError * _Nullable * _Nullable)error
{
  if (_tensor->scalar_type() == torch::executor::ScalarType::Float) {
    const auto *tensorPtr = _tensor->data<float>();
    const auto sizes = _tensor->sizes();
    std::vector<float> tensorVec(tensorPtr, tensorPtr + _tensor->numel());
    std::vector<int32_t> tensorSizes(sizes.begin(), sizes.end());

    NSMutableArray<NSNumber *> *floatArray = [[NSMutableArray alloc] initWithCapacity:tensorVec.size()];
    for (float &i : tensorVec) {
      [floatArray addObject:@(i)];
    }

    NSMutableArray<NSNumber *> *sizesArray = [[NSMutableArray alloc] initWithCapacity:tensorSizes.size()];
    for (int &tensorSize : tensorSizes) {
      [sizesArray addObject:@(tensorSize)];
    }

    return [[ModelRuntimeTensorValueBridgingTuple alloc] initWithFloatArray:floatArray shape:sizesArray];
  }

  if (error) {
    *error = [ModelRuntimeValueErrorFactory
              invalidType:[NSString stringWithFormat:@"torch::executor::ScalarType::%hhd", _tensor->scalar_type()]
              expectedType:@"torch::executor::ScalarType::Float"];
  }

  return nil;
}

- (torch::executor::Tensor)backedValue
{
  return torch::executor::Tensor(_tensor.get());
}

@end
