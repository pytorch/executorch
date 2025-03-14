// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#import <XCTest/XCTest.h>

#import <ExecutorchRuntimeBridge/ExecutorchRuntimeEngine.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExecutorchRuntimeEngineTests : XCTestCase
@end

@implementation ExecutorchRuntimeEngineTests

- (void)testInvalidModel
{
  NSString *const modelPath = @"invalid_model_path";

  NSError *runtimeInitError = nil;
  ExecutorchRuntimeEngine *const engine = [[ExecutorchRuntimeEngine alloc] initWithModelPath:modelPath modelMethodName:@"forward" error:&runtimeInitError];
  XCTAssertNil(engine);
  XCTAssertNotNil(runtimeInitError);

  XCTAssertEqual(runtimeInitError.code, 0);
  XCTAssertEqualObjects(runtimeInitError.userInfo[NSDebugDescriptionErrorKey], @"34");
  // 34 is the code for AccessFailed.
}

- (void)testValidModel
{
  NSBundle *const bundle = [NSBundle bundleForClass:[self class]];
  // This is a simple model that adds two tensors.
  NSString *const modelPath = [bundle pathForResource:@"add" ofType:@"pte"];
  NSError *runtimeInitError = nil;
  ExecutorchRuntimeEngine *const engine = [[ExecutorchRuntimeEngine alloc] initWithModelPath:modelPath modelMethodName:@"forward" error:&runtimeInitError];
  XCTAssertNotNil(engine);
  XCTAssertNil(runtimeInitError);

  ExecutorchRuntimeTensorValue *inputTensor = [[ExecutorchRuntimeTensorValue alloc] initWithFloatArray:@[@2.0] shape:@[@1]];
  ExecutorchRuntimeValue *inputValue = [[ExecutorchRuntimeValue alloc] initWithTensor:inputTensor];

  NSError *inferenceError = nil;
  const auto output = [engine infer:@[inputValue, inputValue] error:&inferenceError];
  XCTAssertNil(inferenceError);

  XCTAssertEqual(output.count, 1);
  NSError *tensorValueError = nil;
  NSError *floatRepresentationError = nil;
  const auto resultTensorValue = [[output.firstObject tensorValueAndReturnError:&tensorValueError]
                                  floatRepresentationAndReturnError:&floatRepresentationError];

  XCTAssertNil(tensorValueError);
  XCTAssertNil(floatRepresentationError);
  XCTAssertEqual(resultTensorValue.floatArray.count, 1);
  XCTAssertEqual(resultTensorValue.shape.count, 1);
  XCTAssertEqual(resultTensorValue.floatArray.firstObject.floatValue, 4.0);
  XCTAssertEqual(resultTensorValue.shape.firstObject.integerValue, 1);
}

@end

NS_ASSUME_NONNULL_END
