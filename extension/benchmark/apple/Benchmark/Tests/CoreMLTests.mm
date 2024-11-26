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
        return [filename hasSuffix:@".mlpackage"];
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
            //      MLModel *model = [MLModel modelWithContentsOfURL:compiledModelURL error:&error];
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.optimizationHints.reshapeFrequency = MLReshapeFrequencyHintFrequent;
            config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
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
                if (error) {
                    XCTFail(@"Failed to create input provider: %@", error.localizedDescription);
                } else {
                    XCTFail(@"Failed with unknown error");
                }
                return;
            }


            MLMultiArray *tokensArray1x1 = [[MLMultiArray alloc] initWithShape:@[@1, @1] dataType:MLMultiArrayDataTypeInt32 error:&error];
            for (NSInteger i = 0; i < tokensArray1x1.count; i++) {
                tokensArray1x1[i] = 0;
            }
            MLDictionaryFeatureProvider *features1x1 = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"tokens": tokensArray1x1} error:&error];


            MLMultiArray *tokensArray1x128 = [[MLMultiArray alloc] initWithShape:@[@1, @128] dataType:MLMultiArrayDataTypeInt32 error:&error];
            for (NSInteger i = 0; i < tokensArray1x128.count; i++) {
                tokensArray1x128[i] = 0;
            }
            MLDictionaryFeatureProvider *features1x128 = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"tokens": tokensArray1x128} error:&error];


            // MLState *state = [model newState];
            [testCase measureWithMetrics:@[[XCTClockMetric new], [XCTMemoryMetric new]]
                                   block:^{
                NSError *error = nil;
                id<MLFeatureProvider> prediction;
                for (int i = 0; i < 50; i++) {
                    // prediction = [model predictionFromFeatures:featureProvider usingState:state error:&error];


                    if (i % 2 == 0) {
                        prediction = [model predictionFromFeatures:features1x128 error:&error];
                    } else {
                        prediction = [model predictionFromFeatures:features1x1 error:&error];
                    }
                    if (error) {
                        XCTFail(@"Prediction failed: %@", error.localizedDescription);
                    }
                }
            }];
        }
    };
}

@end
