/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <XCTest/XCTest.h>

#import "ResourceTestCase.h"

static const int kNumModels = 3;

static MLMultiArray *DummyMultiArrayForFeature(MLFeatureDescription *feature, NSError **error) {
  MLMultiArrayConstraint *constraint = feature.multiArrayConstraint;
  return [[MLMultiArray alloc] initWithShape:constraint.shape
                                    dataType:constraint.dataType
                                       error:error];
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

static MLModel *LoadModelWithFunction(NSURL *compiledModelURL, NSString *functionName, NSError **error) {
  MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
  config.functionName = functionName;
  config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
  return [MLModel modelWithContentsOfURL:compiledModelURL configuration:config error:error];
}

// Returns compiled model URL. If path is already .mlmodelc, returns it directly.
// If path is .mlpackage, compiles it and returns the compiled URL.
static NSURL *GetCompiledModelURL(NSString *modelPath, NSError **error) {
  if ([modelPath hasSuffix:@".mlmodelc"]) {
    // Already compiled
    return [NSURL fileURLWithPath:modelPath];
  } else {
    // Needs compilation
    return [MLModel compileModelAtURL:[NSURL fileURLWithPath:modelPath] error:error];
  }
}

@interface CoreMLTests : ResourceTestCase
@end

@implementation CoreMLTests

+ (NSArray<NSString *> *)directories {
  return @[@"Resources"];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"mod1" : ^BOOL(NSString *filename) {
      return [filename hasSuffix:@"mod1.mlpackage"] || [filename hasSuffix:@"mod1.mlmodelc"];
    },
    @"mod2" : ^BOOL(NSString *filename) {
      return [filename hasSuffix:@"mod2.mlpackage"] || [filename hasSuffix:@"mod2.mlmodelc"];
    },
    @"mod3" : ^BOOL(NSString *filename) {
      return [filename hasSuffix:@"mod3.mlpackage"] || [filename hasSuffix:@"mod3.mlmodelc"];
    }
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:(NSDictionary<NSString *, NSString *> *)resources {
  NSString *mod1Path = resources[@"mod1"];
  NSString *mod2Path = resources[@"mod2"];
  NSString *mod3Path = resources[@"mod3"];

  return @{
    @"multifunction" : ^(XCTestCase *testCase) {
      const BOOL kEnableDecode = YES;
      const BOOL kEnableMod1 = YES;
      const BOOL kEnableMod2 = YES;
      const BOOL kEnableMod3 = YES;

      NSError *error = nil;
      NSArray<NSString *> *allModelPaths = @[mod1Path, mod2Path, mod3Path];
      NSArray<NSString *> *allModelNames = @[@"mod1", @"mod2", @"mod3"];
      NSArray<NSNumber *> *modelEnabled = @[@(kEnableMod1), @(kEnableMod2), @(kEnableMod3)];

      // Filter to only enabled models
      NSMutableArray<NSString *> *modelPaths = [NSMutableArray array];
      NSMutableArray<NSString *> *modelNames = [NSMutableArray array];
      for (int m = 0; m < kNumModels; ++m) {
        if ([modelEnabled[m] boolValue]) {
          [modelPaths addObject:allModelPaths[m]];
          [modelNames addObject:allModelNames[m]];
        }
      }

      const int numEnabledModels = (int)[modelPaths count];
      if (numEnabledModels == 0) {
        XCTFail(@"No models enabled");
        return;
      }

      // Get compiled model URLs (compile if needed)
      NSMutableArray<NSURL *> *compiledModelURLs = [NSMutableArray arrayWithCapacity:numEnabledModels];
      for (int m = 0; m < numEnabledModels; ++m) {
        NSURL *compiledURL = GetCompiledModelURL(modelPaths[m], &error);
        if (error || !compiledURL) {
          XCTFail(@"Failed to get compiled model for %@: %@", modelNames[m], error.localizedDescription);
          return;
        }
        [compiledModelURLs addObject:compiledURL];
      }

      // Load prefill models for enabled models
      NSMutableArray<MLModel *> *prefillModels = [NSMutableArray arrayWithCapacity:numEnabledModels];
      for (int m = 0; m < numEnabledModels; ++m) {
        MLModel *prefillModel = LoadModelWithFunction(compiledModelURLs[m], @"prefill", &error);
        if (error || !prefillModel) {
          XCTFail(@"Failed to load prefill model for %@: %@", modelNames[m], error.localizedDescription);
          return;
        }
        [prefillModels addObject:prefillModel];
      }

      // Load decode models for enabled models if decode is enabled
      NSMutableArray<MLModel *> *decodeModels = [NSMutableArray arrayWithCapacity:numEnabledModels];
      if (kEnableDecode) {
        for (int m = 0; m < numEnabledModels; ++m) {
          MLModel *decodeModel = LoadModelWithFunction(compiledModelURLs[m], @"decode", &error);
          if (error || !decodeModel) {
            XCTFail(@"Failed to load decode model for %@: %@", modelNames[m], error.localizedDescription);
            return;
          }
          [decodeModels addObject:decodeModel];
        }
      }

      // Prepare inputs for prefill models
      NSMutableArray<MLDictionaryFeatureProvider *> *prefillProviders = [NSMutableArray arrayWithCapacity:numEnabledModels];
      for (int m = 0; m < numEnabledModels; ++m) {
        NSMutableDictionary *prefillInputs = DummyInputsForModel(prefillModels[m], &error);
        if (error || !prefillInputs) {
          XCTFail(@"Failed to prepare prefill inputs for %@: %@", modelNames[m], error.localizedDescription);
          return;
        }
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:prefillInputs error:&error];
        if (error || !provider) {
          XCTFail(@"Failed to create prefill input provider for %@: %@", modelNames[m], error.localizedDescription);
          return;
        }
        [prefillProviders addObject:provider];
      }

      // Prepare inputs for decode models if enabled
      NSMutableArray<MLDictionaryFeatureProvider *> *decodeProviders = [NSMutableArray arrayWithCapacity:numEnabledModels];
      if (kEnableDecode) {
        for (int m = 0; m < numEnabledModels; ++m) {
          NSMutableDictionary *decodeInputs = DummyInputsForModel(decodeModels[m], &error);
          if (error || !decodeInputs) {
            XCTFail(@"Failed to prepare decode inputs for %@: %@", modelNames[m], error.localizedDescription);
            return;
          }
          MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:decodeInputs error:&error];
          if (error || !provider) {
            XCTFail(@"Failed to create decode input provider for %@: %@", modelNames[m], error.localizedDescription);
            return;
          }
          [decodeProviders addObject:provider];
        }
      }

      const int kNumPrefillIterations = 30;
      const int kNumDecodeIterations = 50;

      // Start total timing
      CFAbsoluteTime totalStart = CFAbsoluteTimeGetCurrent();

      // Time prefill 1 (call prefill on all enabled models per iteration)
      CFAbsoluteTime prefillStart = CFAbsoluteTimeGetCurrent();
      for (int i = 0; i < kNumPrefillIterations; ++i) {
        for (int m = 0; m < numEnabledModels; ++m) {
          NSError *prefillError = nil;
          id<MLFeatureProvider> prefillPrediction = [prefillModels[m] predictionFromFeatures:prefillProviders[m] error:&prefillError];
          if (prefillError || !prefillPrediction) {
            XCTFail(@"Prefill 1 prediction failed on iteration %d for %@: %@", i, modelNames[m], prefillError.localizedDescription);
            return;
          }
        }
      }
      CFAbsoluteTime prefillEnd = CFAbsoluteTimeGetCurrent();
      double prefillTimeMs = (prefillEnd - prefillStart) * 1000.0;

      // Time decode if enabled (call decode on all enabled models per iteration)
      double decodeTimeMs = 0.0;
      if (kEnableDecode) {
        CFAbsoluteTime decodeStart = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < kNumDecodeIterations; ++i) {
          for (int m = 0; m < numEnabledModels; ++m) {
            NSError *decodeError = nil;
            id<MLFeatureProvider> decodePrediction = [decodeModels[m] predictionFromFeatures:decodeProviders[m] error:&decodeError];
            if (decodeError || !decodePrediction) {
              XCTFail(@"Decode prediction failed on iteration %d for %@: %@", i, modelNames[m], decodeError.localizedDescription);
              return;
            }
          }
        }
        CFAbsoluteTime decodeEnd = CFAbsoluteTimeGetCurrent();
        decodeTimeMs = (decodeEnd - decodeStart) * 1000.0;
      }

      // Time prefill 2 (call prefill on all enabled models per iteration)
      CFAbsoluteTime prefill2Start = CFAbsoluteTimeGetCurrent();
      for (int i = 0; i < kNumPrefillIterations; ++i) {
        for (int m = 0; m < numEnabledModels; ++m) {
          NSError *prefillError = nil;
          id<MLFeatureProvider> prefillPrediction = [prefillModels[m] predictionFromFeatures:prefillProviders[m] error:&prefillError];
          if (prefillError || !prefillPrediction) {
            XCTFail(@"Prefill 2 prediction failed on iteration %d for %@: %@", i, modelNames[m], prefillError.localizedDescription);
            return;
          }
        }
      }
      CFAbsoluteTime prefill2End = CFAbsoluteTimeGetCurrent();
      double prefill2TimeMs = (prefill2End - prefill2Start) * 1000.0;

      // End total timing (includes prefill 1, decode, and prefill 2)
      CFAbsoluteTime totalEnd = CFAbsoluteTimeGetCurrent();
      double totalTimeMs = (totalEnd - totalStart) * 1000.0;

      NSLog(@"=== Benchmark Results ===");
      NSLog(@"Prefill 1: %d iterations x %d models, total time: %.2f ms (%.2f ms/iter)", kNumPrefillIterations, numEnabledModels, prefillTimeMs, prefillTimeMs / kNumPrefillIterations);
      NSLog(@"Decode: %d iterations x %d models, total time: %.2f ms (%.2f ms/iter)", kNumDecodeIterations, numEnabledModels, decodeTimeMs, decodeTimeMs / kNumDecodeIterations);
      NSLog(@"Prefill 2: %d iterations x %d models, total time: %.2f ms (%.2f ms/iter)", kNumPrefillIterations, numEnabledModels, prefill2TimeMs, prefill2TimeMs / kNumPrefillIterations);
      NSLog(@"Total time (prefill 1 + decode + prefill 2): %.2f ms", totalTimeMs);
      NSLog(@"=========================");
    }
  };
}

@end
