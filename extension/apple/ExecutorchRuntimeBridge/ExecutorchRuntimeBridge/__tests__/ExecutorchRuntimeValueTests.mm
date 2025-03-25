/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <XCTest/XCTest.h>

#import <ExecutorchRuntimeBridge/ExecutorchRuntimeValue.h>
#import <executorch/extension/module/module.h>

using torch::executor::EValue;
using torch::executor::TensorImpl;
using torch::executor::ScalarType;

@interface ExecutorchRuntimeValueTests : XCTestCase
@end

@implementation ExecutorchRuntimeValueTests

- (void)testTensorValue
{
  NSMutableArray *data = [NSMutableArray new];
  for (int i = 0; i < 10; i++) {
    [data addObject:@(i + 0.5f)];
  }

  NSArray *shape = @[@(10)];

  ExecutorchRuntimeTensorValue *tensorValue = [[ExecutorchRuntimeTensorValue alloc] initWithFloatArray:data shape:shape];

  const auto floatArray = [tensorValue floatArrayAndReturnError:nil];
  const auto shapeArray = [tensorValue shape];

  XCTAssertEqualObjects(floatArray, data);
  XCTAssertEqualObjects(shapeArray, shape);
}

- (void)testTensorValueWithFloatArrayWithError
{
  std::vector<std::int16_t> data = {1, 2, 3};
  std::vector<int32_t> shape = {3};
  TensorImpl tensorImpl(ScalarType::Int, std::size(shape), shape.data(), data.data());

  XCTAssertNil([[ExecutorchRuntimeTensorValue alloc] initWithTensor:*new torch::executor::Tensor(&tensorImpl) error:nil]);
  NSError *error = nil;
  XCTAssertNil([[ExecutorchRuntimeTensorValue alloc] initWithTensor:*new torch::executor::Tensor(&tensorImpl) error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, static_cast<uint32_t>(executorch::runtime::Error::InvalidArgument));
  XCTAssertEqualObjects(error.userInfo[NSDebugDescriptionErrorKey], @"Invalid type: torch::executor::ScalarType::3, expected torch::executor::ScalarType::Float");
}

- (void)testTensorValueWithError
{
  ExecutorchRuntimeValue *value = [[ExecutorchRuntimeValue alloc] initWithEValue:EValue((int64_t)1)];
  XCTAssertNil([value asTensorValueAndReturnError:nil]);
  NSError *error = nil;
  XCTAssertNil([value asTensorValueAndReturnError:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, static_cast<uint32_t>(executorch::runtime::Error::InvalidArgument));
  XCTAssertEqualObjects(error.userInfo[NSDebugDescriptionErrorKey], @"Invalid type: Tag::4, expected Tag::Tensor");
}

@end
