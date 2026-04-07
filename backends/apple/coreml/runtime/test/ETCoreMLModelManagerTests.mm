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
#import <ETCoreMLModelCache.h>
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

#pragma mark - Cache-based Path Tests

- (void)testModelLoadAndUnloadWithCache {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);

    // Create a separate cache for this test
    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data
                                                    configuration:configuration
                                                       methodName:nil
                                                     functionName:nil
                                                            cache:cache
                                                            error:&localError];
    XCTAssertTrue(handle != NULL, @"Model should load successfully with cache: %@", localError);
    XCTAssertTrue([self.modelManager unloadModelWithHandle:handle]);
    XCTAssertFalse([self.modelManager unloadModelWithHandle:handle]);
}

- (void)testModelHandleWithCache {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);

    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data
                                                    configuration:configuration
                                                       methodName:nil
                                                     functionName:nil
                                                            cache:cache
                                                            error:&localError];
    XCTAssertTrue(handle != NULL, @"Model should load with cache: %@", localError);

    ETCoreMLModel *model = [self.modelManager modelWithHandle:handle];
    XCTAssertNotNil(model.mlModel);
    XCTAssertTrue(model.identifier.length > 0);
    XCTAssertEqual(model.orderedInputNames.count, 2);
    XCTAssertEqual(model.orderedOutputNames.count, 1);
}

- (void)testModelPrewarmWithCache {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);

    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data
                                                    configuration:configuration
                                                       methodName:nil
                                                     functionName:nil
                                                            cache:cache
                                                            error:&localError];
    XCTAssertTrue(handle != NULL, @"Model should load with cache: %@", localError);
    XCTAssertTrue([self.modelManager prewarmModelWithHandle:handle error:&localError], @"Prewarm should succeed: %@", localError);
}

- (void)testAddModelExecutionWithCache {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);

    NSError *localError = nil;
    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data
                                                    configuration:configuration
                                                       methodName:nil
                                                     functionName:nil
                                                            cache:cache
                                                            error:&localError];
    XCTAssertTrue(handle != NULL, @"Model should load with cache: %@", localError);

    ETCoreMLModel *model = [self.modelManager modelWithHandle:handle];
    int x = 20;
    int y = 50;
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

- (void)testMulModelExecutionWithCache {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"bin"];
    XCTAssertNotNil(modelURL);

    NSError *localError = nil;
    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    ModelHandle *handle = [self.modelManager loadModelFromAOTData:data
                                                    configuration:configuration
                                                       methodName:nil
                                                     functionName:nil
                                                            cache:cache
                                                            error:&localError];
    XCTAssertTrue(handle != NULL, @"Model should load with cache: %@", localError);

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

- (void)testCacheHitOnReload {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSError *localError = nil;
    XCTAssertNotNil(modelURL);

    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *data = [NSData dataWithContentsOfURL:modelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    // Load model first time (cold cache)
    ModelHandle *handle1 = [self.modelManager loadModelFromAOTData:data
                                                     configuration:configuration
                                                        methodName:nil
                                                      functionName:nil
                                                             cache:cache
                                                             error:&localError];
    XCTAssertTrue(handle1 != NULL, @"First load should succeed: %@", localError);

    ETCoreMLModel *model1 = [self.modelManager modelWithHandle:handle1];
    NSString *identifier = model1.identifier;

    // Unload
    XCTAssertTrue([self.modelManager unloadModelWithHandle:handle1]);

    // Load model second time (should hit cache)
    ModelHandle *handle2 = [self.modelManager loadModelFromAOTData:data
                                                     configuration:configuration
                                                        methodName:nil
                                                      functionName:nil
                                                             cache:cache
                                                             error:&localError];
    XCTAssertTrue(handle2 != NULL, @"Second load should succeed from cache: %@", localError);

    ETCoreMLModel *model2 = [self.modelManager modelWithHandle:handle2];
    XCTAssertEqualObjects(model2.identifier, identifier, @"Identifier should match");

    // Verify model still works
    int x = 10;
    int y = 20;
    NSArray<MLMultiArray *> *inputs = [ETCoreMLTestUtils inputsForModel:model2 repeatedValues:@[@(x), @(y)] error:&localError];
    XCTAssertNotNil(inputs);
    MLMultiArray *output = [ETCoreMLTestUtils filledMultiArrayWithShape:inputs[0].shape dataType:inputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *args = [inputs arrayByAddingObject:output];

    XCTAssertTrue([self.modelManager executeModelWithHandle:handle2
                                                       args:args
                                             loggingOptions:executorchcoreml::ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError]);
    for (NSUInteger i = 0; i < output.count; i++) {
        NSNumber *value = [output objectAtIndexedSubscript:i];
        XCTAssertEqual(value.integerValue, x + y);
    }
}

- (void)testMultipleModelsWithSameCache {
    NSURL *addModelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"bin"];
    NSURL *mulModelURL = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"bin"];
    XCTAssertNotNil(addModelURL);
    XCTAssertNotNil(mulModelURL);

    NSError *localError = nil;
    NSURL *cacheURL = [self.testDirectoryURL URLByAppendingPathComponent:@"model_cache"];
    ETCoreMLModelCache *cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
    XCTAssertTrue(cache.isReady, @"Cache should be ready: %@", cache.initializationError);

    NSData *addData = [NSData dataWithContentsOfURL:addModelURL];
    NSData *mulData = [NSData dataWithContentsOfURL:mulModelURL];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsAll;

    // Load both models using the same cache
    ModelHandle *addHandle = [self.modelManager loadModelFromAOTData:addData
                                                       configuration:configuration
                                                          methodName:nil
                                                        functionName:nil
                                                               cache:cache
                                                               error:&localError];
    XCTAssertTrue(addHandle != NULL, @"Add model should load: %@", localError);

    ModelHandle *mulHandle = [self.modelManager loadModelFromAOTData:mulData
                                                       configuration:configuration
                                                          methodName:nil
                                                        functionName:nil
                                                               cache:cache
                                                               error:&localError];
    XCTAssertTrue(mulHandle != NULL, @"Mul model should load: %@", localError);

    // Verify both models work correctly
    ETCoreMLModel *addModel = [self.modelManager modelWithHandle:addHandle];
    ETCoreMLModel *mulModel = [self.modelManager modelWithHandle:mulHandle];

    int x = 5;
    int y = 3;

    // Test add model
    NSArray<MLMultiArray *> *addInputs = [ETCoreMLTestUtils inputsForModel:addModel repeatedValues:@[@(x), @(y)] error:&localError];
    MLMultiArray *addOutput = [ETCoreMLTestUtils filledMultiArrayWithShape:addInputs[0].shape dataType:addInputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *addArgs = [addInputs arrayByAddingObject:addOutput];

    XCTAssertTrue([self.modelManager executeModelWithHandle:addHandle
                                                       args:addArgs
                                             loggingOptions:executorchcoreml::ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError]);
    XCTAssertEqual([addOutput objectAtIndexedSubscript:0].integerValue, x + y);

    // Test mul model
    NSArray<MLMultiArray *> *mulInputs = [ETCoreMLTestUtils inputsForModel:mulModel repeatedValues:@[@(x), @(y)] error:&localError];
    MLMultiArray *mulOutput = [ETCoreMLTestUtils filledMultiArrayWithShape:mulInputs[0].shape dataType:mulInputs[0].dataType repeatedValue:@(0) error:&localError];
    NSArray<MLMultiArray *> *mulArgs = [mulInputs arrayByAddingObject:mulOutput];

    XCTAssertTrue([self.modelManager executeModelWithHandle:mulHandle
                                                       args:mulArgs
                                             loggingOptions:executorchcoreml::ModelLoggingOptions()
                                                eventLogger:nullptr
                                                      error:&localError]);
    XCTAssertEqual([mulOutput objectAtIndexedSubscript:0].integerValue, x * y);
}

#pragma mark - Autorelease Pool Tests

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
