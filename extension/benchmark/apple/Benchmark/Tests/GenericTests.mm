/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResourceTestCase.h"

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

#define ASSERT_OK_OR_RETURN(value__)            \
  ({                                            \
    XCTAssertEqual(value__.error(), Error::Ok); \
    if (!value__.ok()) {                        \
      return;                                   \
    }                                           \
  })

@interface GenericTests : ResourceTestCase
@end

@implementation GenericTests

+ (NSArray<NSString *> *)directories {
  return @[
    @"Resources",
    @"aatp/data", // AWS Farm devices look for resources here.
  ];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"model" : ^BOOL(NSString *filename){
      return [filename hasSuffix:@".pte"];
    },
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:
    (NSDictionary<NSString *, NSString *> *)resources {
  NSString *modelPath = resources[@"model"];
  return @{
    @"load" : ^(XCTestCase *testCase){
      [testCase
          measureWithMetrics:@[ [XCTClockMetric new], [XCTMemoryMetric new] ]
                       block:^{
                         XCTAssertEqual(
                             Module(modelPath.UTF8String).load_forward(),
                             Error::Ok);
                       }];
    },
    @"forward" : ^(XCTestCase *testCase) {
      auto __block module = std::make_unique<Module>(modelPath.UTF8String);

      const auto method_meta = module->method_meta("forward");
      ASSERT_OK_OR_RETURN(method_meta);

      const auto num_inputs = method_meta->num_inputs();
      XCTAssertGreaterThan(num_inputs, 0);

      std::vector<TensorPtr> tensors;
      tensors.reserve(num_inputs);

      for (auto index = 0; index < num_inputs; ++index) {
        const auto input_tag = method_meta->input_tag(index);
        ASSERT_OK_OR_RETURN(input_tag);

        switch (*input_tag) {
        case Tag::Tensor: {
          const auto tensor_meta = method_meta->input_tensor_meta(index);
          ASSERT_OK_OR_RETURN(tensor_meta);

          const auto sizes = tensor_meta->sizes();
          tensors.emplace_back(
              rand({sizes.begin(), sizes.end()}, tensor_meta->scalar_type()));
          XCTAssertEqual(module->set_input(tensors.back(), index), Error::Ok);
        } break;
        default:
          XCTFail("Unsupported tag %i at input %d", *input_tag, index);
        }
      }
      XCTMeasureOptions *options = [[XCTMeasureOptions alloc] init];
      options.iterationCount = 20;
      [testCase measureWithMetrics:@[ [XCTClockMetric new], [XCTMemoryMetric new] ]
                            options:options
                            block:^{
                              XCTAssertEqual(module->forward().error(), Error::Ok);
                            }];
    },
  };
}

@end
