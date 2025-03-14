// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#import <XCTest/XCTest.h>

#import <ExecutorchRuntimeBridge/ExecutorchRuntimeValue.h>
#import <ModelRunnerDataKit/ModelRunnerDataKit-Swift.h>
#import <executorch/extension/module/module.h>

using torch::executor::EValue;
using torch::executor::TensorImpl;
using torch::executor::ScalarType;

@interface ExecutorchRuntimeValueTests : XCTestCase
@end

@implementation ExecutorchRuntimeValueTests

- (void)testStringValueWithError
{
  ExecutorchRuntimeValue *value = [[ExecutorchRuntimeValue alloc] initWithEValue:EValue((int64_t)1)];
  XCTAssertNil([value stringValueAndReturnError:nil]);
  NSError *error = nil;
  XCTAssertNil([value stringValueAndReturnError:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqualObjects([error description], @"Unsupported type: ExecutorchRuntimeValue doesn't support strings");
}

- (void)testTensorValue
{
  NSMutableArray *data = [NSMutableArray new];
  for (int i = 0; i < 10; i++) {
    [data addObject:@(i + 0.5f)];
  }

  NSArray *shape = @[@(10)];

  ExecutorchRuntimeTensorValue *tensorValue = [[ExecutorchRuntimeTensorValue alloc] initWithFloatArray:data shape:shape];

  const auto tuple = [tensorValue floatRepresentationAndReturnError:nil];
  XCTAssertEqualObjects(tuple.floatArray, data);
  XCTAssertEqualObjects(tuple.shape, shape);
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
  XCTAssertEqualObjects([error description], @"Invalid type: torch::executor::ScalarType::3, expected torch::executor::ScalarType::Float");
}

- (void)testTensorValueWithError
{
  ExecutorchRuntimeValue *value = [[ExecutorchRuntimeValue alloc] initWithEValue:EValue((int64_t)1)];
  XCTAssertNil([value tensorValueAndReturnError:nil]);
  NSError *error = nil;
  XCTAssertNil([value tensorValueAndReturnError:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqualObjects([error description], @"Invalid type: Tag::4, expected Tag::Tensor");
}

@end
