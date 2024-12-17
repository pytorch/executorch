/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResourceTestCase.h"

#import <CoreML/CoreML.h>

static MLMultiArray *DummyMultiArrayForFeature(MLFeatureDescription *feature, NSError **error) {
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:feature.multiArrayConstraint.shape
                                                     dataType:feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeInt32 ? MLMultiArrayDataTypeInt32 : MLMultiArrayDataTypeDouble
                                                        error:error];
    for (auto index = 0; index < array.count; ++index) {
        array[index] = feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeInt32 ? @1 : @1.0;
    }
    return array;
}

static NSMutableDictionary *DummyInputsForModel(MLModel *model, NSError **error) {
    NSMutableDictionary *inputs = [NSMutableDictionary dictionary];
    NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = model.modelDescription.inputDescriptionsByName;

    for (NSString *inputName in inputDescriptions) {
        MLFeatureDescription *feature = inputDescriptions[inputName];

        switch (feature.type) {
            case MLFeatureTypeMultiArray: {
                MLMultiArray *array = DummyMultiArrayForFeature(feature, error);
                inputs[inputName] = [MLFeatureValue featureValueWithMultiArray:array];
                break;
            }
            case MLFeatureTypeInt64:
                inputs[inputName] = [MLFeatureValue featureValueWithInt64:1];
                break;
            case MLFeatureTypeDouble:
                inputs[inputName] = [MLFeatureValue featureValueWithDouble:1.0];
                break;
            case MLFeatureTypeString:
                inputs[inputName] = [MLFeatureValue featureValueWithString:@"1"];
                break;
            default:
                break;
        }
    }
    return inputs;
}

@interface CoreMLTests : ResourceTestCase
@end

@implementation CoreMLTests

+ (NSArray<NSString *> *)directories {
    return @[@"Resources"];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
    return @{ @"model" : ^BOOL(NSString *filename) {
        return [filename hasSuffix:@"combined.mlpackage"];
    }};
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:(NSDictionary<NSString *, NSString *> *)resources {
    NSString *modelPath = resources[@"model"];

    return @{
        @"prediction" : ^(XCTestCase *testCase) {

            NSError *error = nil;
            NSURL *compiledModelURL = [MLModel compileModelAtURL:[NSURL fileURLWithPath:modelPath] error:&error];
            if (error || !compiledModelURL) {
                XCTFail(@"Failed to compile model: %@", error.localizedDescription);
                return;
            }

            // Model1
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            config.functionName = @"model1";
            MLModel *model = [MLModel modelWithContentsOfURL:compiledModelURL configuration:config error:&error];
            if (error || !model) {
                XCTFail(@"Failed to load model: %@", error.localizedDescription);
                return;
            }
            NSMutableDictionary *inputs = DummyInputsForModel(model, &error);
            if (error || !inputs) {
                XCTFail(@"Failed to prepare inputs: %@", error.localizedDescription);
                return;
            }
            MLDictionaryFeatureProvider *featureProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&error];
            if (error || !featureProvider) {
                XCTFail(@"Failed to create input provider: %@", error.localizedDescription);
                return;
            }


            // Model2
            MLModelConfiguration *config2 = [[MLModelConfiguration alloc] init];
            config2.computeUnits = MLComputeUnitsCPUOnly;
            config2.functionName = @"model2";
            MLModel *model2 = [MLModel modelWithContentsOfURL:compiledModelURL configuration:config2 error:&error];
            if (error || !model2) {
                XCTFail(@"Failed to load model: %@", error.localizedDescription);
                return;
            }
            NSMutableDictionary *inputs2 = DummyInputsForModel(model2, &error);
            if (error || !inputs2) {
                XCTFail(@"Failed to prepare inputs: %@", error.localizedDescription);
                return;
            }
            MLDictionaryFeatureProvider *featureProvider2 = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs2 error:&error];
            if (error || !featureProvider) {
                XCTFail(@"Failed to create input provider: %@", error.localizedDescription);
                return;
            }


            MLState *state = [model2 newState];
            [testCase measureWithMetrics:@[[XCTClockMetric new], [XCTMemoryMetric new]]
                                   block:^{
                NSError *error = nil;

                // Prefill
                id<MLFeatureProvider> prediction;
                prediction = [model predictionFromFeatures:featureProvider error:&error];
                if (error) {
                    XCTFail(@"Prediction failed: %@", error.localizedDescription);
                }

                // Decode
                id<MLFeatureProvider> prediction2;
                for (int i = 0; i < 128; i++) {
                    prediction2 = [model2 predictionFromFeatures:featureProvider2 usingState:state error:&error];
                    if (error) {
                        XCTFail(@"Prediction failed: %@", error.localizedDescription);
                    }
                }
            }];
        }
    };
}

@end
