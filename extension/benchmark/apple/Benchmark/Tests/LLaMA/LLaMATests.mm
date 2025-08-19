/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResourceTestCase.h"

#import <ExecuTorchLLM/ExecuTorchLLM.h>

@interface TokensPerSecondMetric : NSObject<XCTMetric>

@property(nonatomic, assign) NSUInteger tokenCount;

@end

@implementation TokensPerSecondMetric

- (id)copyWithZone:(NSZone *)zone {
  TokensPerSecondMetric *copy = [[[self class] allocWithZone:zone] init];
  copy.tokenCount = self.tokenCount;
  return copy;
}

- (NSArray<XCTPerformanceMeasurement *> *)
    reportMeasurementsFromStartTime:
        (XCTPerformanceMeasurementTimestamp *)startTime
                          toEndTime:
                              (XCTPerformanceMeasurementTimestamp *)endTime
                              error:(NSError **)error {
  double elapsedTime =
      (endTime.absoluteTimeNanoSeconds - startTime.absoluteTimeNanoSeconds) /
      (double)NSEC_PER_SEC;
  return @[ [[XCTPerformanceMeasurement alloc]
      initWithIdentifier:NSStringFromClass([self class])
             displayName:@"Tokens Per Second"
             doubleValue:(self.tokenCount / elapsedTime)
              unitSymbol:@"t/s"] ];
}

@end

@interface LLaMATests : ResourceTestCase
@end

@implementation LLaMATests

+ (NSArray<NSString *> *)directories {
  return @[
    @"Resources",
    @"aatp/data", // AWS Farm devices look for resources here.
  ];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"model" : ^BOOL(NSString *filename){
      return [filename hasSuffix:@".pte"] && [filename.lowercaseString containsString:@"llm"];
    },
    @"tokenizer" : ^BOOL(NSString *filename) {
      return [filename isEqual:@"tokenizer.bin"] || [filename isEqual:@"tokenizer.model"] || [filename isEqual:@"tokenizer.json"];
    },
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:
    (NSDictionary<NSString *, NSString *> *)resources {
  NSString *modelPath = resources[@"model"];
  NSString *tokenizerPath = resources[@"tokenizer"];
  return @{
    @"generate" : ^(XCTestCase *testCase){
      NSMutableArray<NSString *> *specialTokens = [@[
        @"<|begin_of_text|>",
        @"<|end_of_text|>",
        @"<|reserved_special_token_0|>",
        @"<|reserved_special_token_1|>",
        @"<|finetune_right_pad_id|>",
        @"<|step_id|>",
        @"<|start_header_id|>",
        @"<|end_header_id|>",
        @"<|eom_id|>",
        @"<|eot_id|>",
        @"<|python_tag|>"
      ] mutableCopy];
      for (NSUInteger index = 2; specialTokens.count < 256; ++index) {
        [specialTokens addObject:[NSString stringWithFormat:@"<|reserved_special_token_%zu|>", index]];
      }
      auto __block runner = [[ExecuTorchTextLLMRunner alloc] initWithModelPath:modelPath
                                                                 tokenizerPath:tokenizerPath
                                                                 specialTokens:specialTokens];
      NSError *error;
      BOOL status = [runner loadWithError:&error];
      if (!status) {
        XCTFail("Load failed with error %zi", error.code);
        return;
      }
      TokensPerSecondMetric *tokensPerSecondMetric = [TokensPerSecondMetric new];
      [testCase measureWithMetrics:@[ tokensPerSecondMetric, [XCTClockMetric new], [XCTMemoryMetric new] ]
                            block:^{
                              tokensPerSecondMetric.tokenCount = 0;
                              BOOL status = [runner generate:@"Once upon a time"
                                              sequenceLength:50
                                           withTokenCallback:^(NSString *token) {
                                tokensPerSecondMetric.tokenCount++;
                              }
                                                       error:NULL];
                              XCTAssertTrue(status);
                            }];
    },
  };
}

@end
