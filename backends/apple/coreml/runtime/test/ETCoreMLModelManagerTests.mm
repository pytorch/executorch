//
// ETCoreMLModelManagerTests.m
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"
#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelManager.h>
#import <MLModel_Prewarm.h>
#import <XCTest/XCTest.h>
#import <executorch/runtime/platform/runtime.h>
#import <model_logging_options.h>
#import <multiarray.h>

using namespace executorchcoreml;

@interface ETCoreMLModelManagerTests : XCTestCase

@property (strong, nonatomic, nullable) ETCoreMLModelManager *modelManager;
@property (strong, nonatomic, nullable) NSFileManager *fileManager;
@property (copy, nonatomic) NSURL *testDirectoryURL;

@end

@implementation ETCoreMLModelManagerTests

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:ETCoreMLModelManagerTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

- (void)setUp {
    executorch::runtime::runtime_init();
    @autoreleasepool {
        NSError *localError = nil;
        self.fileManager = [[NSFileManager alloc] init];
        self.testDirectoryURL = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString]];
        [self.fileManager removeItemAtURL:self.testDirectoryURL error:&localError];
        ETCoreMLAssetManager *assetManager = [ETCoreMLTestUtils createAssetManagerWithURL:self.testDirectoryURL error:&localError];
        XCTAssertNotNil(assetManager);
        self.modelManager = [[ETCoreMLModelManager alloc] initWithAssetManager:assetManager];
    }
}

- (void)tearDown {
    @autoreleasepool {
        NSError *localError = nil;
        self.modelManager = nil;
        XCTAssertTrue([self.fileManager removeItemAtURL:self.testDirectoryURL error:&localError]);
        self.fileManager = nil;
    }
}

- (void)testModelLoadAndUnload {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data configuration:configuration error:&localError];
    XCTAssertTrue([self.modelManager unloadModelWithHandle:handle]);
    XCTAssertFalse([self.modelManager unloadModelWithHandle:handle]);
}

- (void)testModelHandle {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data configuration:configuration error:&localError];
    ETCoreMLModel *model = [self.modelManager modelWithHandle:handle];
    XCTAssertNotNil(model.mlModel);
    XCTAssertTrue(model.identifier.length > 0);
    XCTAssertEqual(model.orderedInputNames.count, 2);
    XCTAssertEqual(model.orderedOutputNames.count, 1);
}

- (void)testModelPrewarm {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data configuration:configuration error:&localError];
    XCTAssertTrue([self.modelManager prewarmModelWithHandle:handle error:&localError]);
}

- (void)testAddModelExecution {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);
    
    NSError *localError = nil;
    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data configuration:configuration error:&localError];
    ETCoreMLModel *model = [self.modelManager modelWithHandle:handle];
    int x = 20;
    int y = 50;
    // add_coreml_all does the following operation.
    int z = x + y;
    
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];
    XCTAssertTrue([self.modelManager executeModelWithHandle:handle
                                                       args:args
                                             loggingOptions:executorchcoreml::ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError]);
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
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data configuration:configuration error:&localError];
    ETCoreMLModel *model = [self.modelManager modelWithHandle:handle];
    int x = 20;
    int y = 50;
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];
    XCTAssertTrue([self.modelManager executeModelWithHandle:handle
                                                       args:args
                                            loggingOptions:executorchcoreml::ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError]);
    for (NSUInteger i = 0; i < output.count; i++) {
        NSNumber *value = [output objectAtIndexedSubscript:i];
        XCTAssertEqual(value.integerValue, x * y);
    }
}

// See https://github.com/pytorch/executorch/pull/10465
- (void)testAutoreleasepoolError {
    NSURL *modelURL = [self.class bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);

    NSData *modelData = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;
    ModelHandle *modelHandle = [self.modelManager loadModelFromAOTData:modelData
                                                           configuration:configuration
                                                                   error:&localError];
    XCTAssert(modelHandle);

    ETCoreMLModel *model = [self.modelManager modelWithHandle:modelHandle];
    XCTAssert(model);

    NSArray<MLMultiArray *> *inputArrays =
        [ETCoreMLTestUtils inputsForModel:model repeatedValues:@[@(2), @(3)] error:&localError];
    XCTAssert(inputArrays);

    std::vector<MultiArray> multiArrays;
    multiArrays.reserve(inputArrays.count + model.orderedOutputNames.count);
    for (MLMultiArray *array in inputArrays) {
        auto dataTypeOpt = to_multiarray_data_type(array.dataType);
        XCTAssert(dataTypeOpt.has_value());
        auto dataType = dataTypeOpt.value();

        std::vector<size_t> dims;
        for (NSNumber *n in array.shape) {
            dims.push_back(n.unsignedLongValue);
        }

        std::vector<ssize_t> strides(dims.size());
        ssize_t currentStride = 1;
        for (NSInteger i = dims.size() - 1; i >= 0; --i) {
            strides[i] = currentStride;
            currentStride *= dims[i];
        }

        multiArrays.emplace_back(array.dataPointer,
                                 MultiArray::MemoryLayout(dataType, dims, strides));
    }

    auto inputLayout = multiArrays[0].layout();
    size_t bufferSize = inputLayout.num_bytes();
    for (NSUInteger i = 0; i < model.orderedOutputNames.count; ++i) {
        multiArrays.emplace_back(calloc(1, bufferSize), inputLayout);
    }
    // corrupt first input shape to force error
    {
        auto originalLayout = multiArrays[0].layout();
        auto corruptedDims = originalLayout.shape();
        corruptedDims[0] += 1;
        multiArrays[0] = MultiArray(multiArrays[0].data(),
                                    MultiArray::MemoryLayout(originalLayout.dataType(),
                                                             corruptedDims,
                                                             originalLayout.strides()));
    }

    BOOL success = [self.modelManager executeModelWithHandle:modelHandle
                                                    argsVec:multiArrays
                                             loggingOptions:ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError];
    XCTAssertFalse(success);
    XCTAssertNotNil(localError);

    for (size_t i = inputArrays.count; i < multiArrays.size(); ++i) {
        free(multiArrays[i].data());
    }
}

@end
