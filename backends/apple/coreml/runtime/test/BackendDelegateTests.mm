//
// BackendDelegateTests.m
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <CoreML/CoreML.h>

#import <backend_delegate.h>
#import <coreml_backend/delegate.h>
#import <multiarray.h>

#import <ETCoreMLModel.h>
#import <ETCoreMLStrings.h>

#import "ETCoreMLTestUtils.h"

using namespace executorchcoreml;

namespace {
template<typename T>
T toValue(NSNumber *value);

template<>
size_t toValue(NSNumber *value) {
    return value.unsignedLongLongValue;
}

template<>
ssize_t toValue(NSNumber *value) {
    return value.longLongValue;
}

template<typename T>
std::vector<T> toVector(NSArray<NSNumber *> *values) {
    std::vector<T> result;
    result.reserve(values.count);
    for (NSNumber *value in values) {
        result.emplace_back(toValue<T>(value));
    }
    
    return result;
}

MultiArray::DataType toDataType(MLMultiArrayDataType dataType) {
    switch (dataType) {
        case MLMultiArrayDataTypeFloat: {
            return MultiArray::DataType::Float;
        }
        case MLMultiArrayDataTypeFloat16: {
            return MultiArray::DataType::Float16;
        }
        case MLMultiArrayDataTypeDouble: {
            return MultiArray::DataType::Double;
        }
        case MLMultiArrayDataTypeInt32: {
            return MultiArray::DataType::Int;
        }
    }
}

MultiArray toMultiArray(MLMultiArray *mlMultiArray) {
    auto shape = toVector<size_t>(mlMultiArray.shape);
    auto strides = toVector<ssize_t>(mlMultiArray.strides);
    auto layout = MultiArray::MemoryLayout(toDataType(mlMultiArray.dataType), std::move(shape), std::move(strides));
    __block void *bytes = nullptr;
    [mlMultiArray getMutableBytesWithHandler:^(void *mutableBytes, __unused NSInteger size, __unused NSArray<NSNumber *> *strides) {
        bytes = mutableBytes;
    }];
    
    return MultiArray(bytes, std::move(layout));
}

std::vector<MultiArray> toMultiArrays(NSArray<MLMultiArray *> *mlMultiArrays) {
    std::vector<MultiArray> result;
    result.reserve(mlMultiArrays.count);
    
    for (MLMultiArray *mlMultiArray in mlMultiArrays) {
        result.emplace_back(toMultiArray(mlMultiArray));
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
        NSData *specData = [@"cpu_and_ane" dataUsingEncoding:NSUTF8StringEncoding];
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
    z = z + x;
    z = z + x;
    z = z + z;
    
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];
    std::error_code errorCode;
    XCTAssertTrue(_delegate->execute(handle, toMultiArrays(args), errorCode));
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
    XCTAssertTrue(_delegate->execute(handle, toMultiArrays(args), errorCode));
    for (NSUInteger i = 0; i < output.count; i++) {
        NSNumber *value = [output objectAtIndexedSubscript:i];
        XCTAssertEqual(value.integerValue, x * y);
    }
}

@end
