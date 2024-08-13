//
// BackendDelegateTests.m
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"
#import <CoreML/CoreML.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLStrings.h>
#import <XCTest/XCTest.h>
#import <backend_delegate.h>
#import <coreml_backend/delegate.h>
#import <model_logging_options.h>
#import <multiarray.h>
#import <objc_array_util.h>
#import <executorch/runtime/platform/runtime.h>

using namespace executorchcoreml;

namespace {

MultiArray to_multiarray(MLMultiArray *ml_multiarray) {
    auto shape = to_vector<size_t>(ml_multiarray.shape);
    auto strides = to_vector<ssize_t>(ml_multiarray.strides);
    auto layout = MultiArray::MemoryLayout(to_multiarray_data_type(ml_multiarray.dataType).value(),
                                           std::move(shape),
                                           std::move(strides));
    __block void *bytes = nullptr;
    [ml_multiarray getMutableBytesWithHandler:^(void *mutableBytes, __unused NSInteger size, __unused NSArray<NSNumber *> *strides) {
        bytes = mutableBytes;
    }];
    
    return MultiArray(bytes, std::move(layout));
}

std::vector<MultiArray> to_multiarrays(NSArray<MLMultiArray *> *ml_multiarrays) {
    std::vector<MultiArray> result;
    result.reserve(ml_multiarrays.count);
    
    for (MLMultiArray *ml_multiarray in ml_multiarrays) {
        result.emplace_back(to_multiarray(ml_multiarray));
    }
    return result;
}
}

@interface BackendDelegateTests : XCTestCase

@end

@implementation BackendDelegateTests {
    std::shared_ptr<BackendDelegate> _delegate;
}

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:BackendDelegateTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

+ (void)setUp {
    torch::executor::runtime_init();
}

- (void)setUp {
    @autoreleasepool {
        _delegate = BackendDelegate::make(BackendDelegate::Config());
    }
}

- (void)tearDown {
    @autoreleasepool {
        _delegate.reset();
    }
}

- (void)testDelegateAvailability {
    XCTAssertTrue(_delegate->is_available());
}

- (void)testDelegateInit {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), {});
    XCTAssert(handle != nullptr);
}

- (void)testCompileSpecs {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    std::string computeUnitsKey(ETCoreMLStrings.computeUnitsKeyName.UTF8String);
    {
        std::unordered_map<std::string, Buffer> compileSpecs;
        NSData *specData = [@"cpu_only" dataUsingEncoding:NSUTF8StringEncoding];
        compileSpecs.emplace(computeUnitsKey, Buffer(specData.bytes, specData.length));
        BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), compileSpecs);
        ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
        XCTAssert(model.mlModel.configuration.computeUnits == MLComputeUnitsCPUOnly);
        _delegate->destroy(handle);
    }
    
    {
        std::unordered_map<std::string, Buffer> compileSpecs;
        NSData *specData = [@"cpu_and_gpu" dataUsingEncoding:NSUTF8StringEncoding];
        compileSpecs.emplace(computeUnitsKey, Buffer(specData.bytes, specData.length));
        BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), compileSpecs);
        ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
        XCTAssert(model.mlModel.configuration.computeUnits == MLComputeUnitsCPUAndGPU);
        _delegate->destroy(handle);
    }
    
    {
        std::unordered_map<std::string, Buffer> compileSpecs;
        NSData *specData = [@"cpu_and_ne" dataUsingEncoding:NSUTF8StringEncoding];
        compileSpecs.emplace(computeUnitsKey, Buffer(specData.bytes, specData.length));
        BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), compileSpecs);
        ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
        XCTAssert(model.mlModel.configuration.computeUnits == MLComputeUnitsCPUAndNeuralEngine);
        _delegate->destroy(handle);
    }
}

- (void)testDelegateDestroy {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), {});
    XCTAssert(handle != nullptr);
    XCTAssertTrue(_delegate->is_valid_handle(handle));
    _delegate->destroy(handle);
    XCTAssertFalse(_delegate->is_valid_handle(handle));
}

- (void)testDelegateNumArgs {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), {});
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
    auto pair = _delegate->get_num_arguments(handle);
    XCTAssertEqual(model.orderedInputNames.count, pair.first);
    XCTAssertEqual(model.orderedOutputNames.count, pair.second);
}

- (void)testAddModelExecution {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSError *localError = nil;
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), {});
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
    int x = 20;
    int y = 50;
    // add_coreml_all does the following operations.
    int z = x + y;
    
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];
    std::error_code errorCode;
    XCTAssertTrue(_delegate->execute(handle,
                                     to_multiarrays(args),
                                     ModelLoggingOptions(),
                                     nullptr,
                                     errorCode));
    for (NSUInteger i = 0; i < output.count; i++) {
        NSNumber *value = [output objectAtIndexedSubscript:i];
        XCTAssertEqual(value.integerValue, z);
    }
}

- (void)testMulModelExecution {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    NSError *localError = nil;
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    BackendDelegate::Handle *handle = _delegate->init(Buffer(data.bytes, data.length), {});
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
    int x = 20;
    int y = 50;
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];
    std::error_code errorCode;
    XCTAssertTrue(_delegate->execute(handle, 
                                     to_multiarrays(args),
                                     ModelLoggingOptions(),
                                     nullptr,
                                     errorCode));
    for (NSUInteger i = 0; i < output.count; i++) {
        NSNumber *value = [output objectAtIndexedSubscript:i];
        XCTAssertEqual(value.integerValue, x * y);
    }
}

@end
