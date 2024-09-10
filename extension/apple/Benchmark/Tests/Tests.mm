/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <XCTest/XCTest.h>

#import <objc/runtime.h>

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

@interface Tests : XCTestCase
@end

@implementation Tests

+ (void)initialize {
  if (self != [self class]) {
    return;
  }
  for (NSBundle *bundle in @[
         [NSBundle mainBundle],
         [NSBundle bundleForClass:[self class]],
       ]) {
    for (NSString *directory in @[
           @"Models",
           @"aatp/data",
         ]) {
      NSString *directoryPath =
          [bundle.resourcePath stringByAppendingPathComponent:directory];
      NSArray *filePaths =
          [NSFileManager.defaultManager contentsOfDirectoryAtPath:directoryPath
                                                            error:nil];
      for (NSString *filePath in filePaths) {
        if (![filePath hasSuffix:@".pte"]) {
          continue;
        }
        NSString *modelPath =
            [directoryPath stringByAppendingPathComponent:filePath];
        NSString *directoryName =
            [directory stringByReplacingOccurrencesOfString:@"/"
                                                 withString:@"_"]
                .lowercaseString;
        NSString *modelName =
            modelPath.lastPathComponent.stringByDeletingPathExtension;

        SEL testLoadSelector = NSSelectorFromString([NSString
            stringWithFormat:@"test_load_%@_%@", directoryName, modelName]);
        IMP testLoadImplementation = imp_implementationWithBlock(^(id _self) {
          auto __block module = std::make_unique<Module>(modelPath.UTF8String);
          [_self measureWithMetrics:@[
            [XCTClockMetric new],
            [XCTMemoryMetric new],
          ]
                            options:XCTMeasureOptions.defaultOptions
                              block:^{
                                XCTAssertEqual(module->load_method("forward"),
                                               Error::Ok);
                              }];
        });
        class_addMethod(
            [self class], testLoadSelector, testLoadImplementation, "v@:");

        SEL testForwardSelector = NSSelectorFromString([NSString
            stringWithFormat:@"test_forward_%@_%@", directoryName, modelName]);
        IMP testForwardImplementation = imp_implementationWithBlock(^(
            id _self) {
          auto __block module = std::make_unique<Module>(modelPath.UTF8String);
          XCTAssertEqual(module->load_method("forward"), Error::Ok);

          const auto method_meta = module->method_meta("forward");
          XCTAssertEqual(method_meta.error(), Error::Ok);

          const auto num_inputs = method_meta->num_inputs();
          XCTAssertGreaterThan(num_inputs, 0);

          std::vector<TensorPtr> __block tensors;
          tensors.reserve(num_inputs);
          std::vector<EValue> __block inputs;
          inputs.reserve(num_inputs);

          for (auto index = 0; index < num_inputs; ++index) {
            const auto input_tag = method_meta->input_tag(index);
            XCTAssertEqual(input_tag.error(), Error::Ok);

            switch (*input_tag) {
            case Tag::Tensor: {
              const auto tensor_meta = method_meta->input_tensor_meta(index);
              XCTAssertEqual(tensor_meta.error(), Error::Ok);

              const auto sizes = tensor_meta->sizes();
              tensors.emplace_back(make_tensor_ptr(
                  tensor_meta->scalar_type(),
                  {sizes.begin(), sizes.end()},
                  std::vector<uint8_t>(tensor_meta->nbytes(), 0b01010101)));
              inputs.emplace_back(tensors.back());
            } break;
            default:
              XCTFail("Unsupported tag %i at input %d", *input_tag, index);
            }
          }
          [_self measureWithMetrics:@[
            [XCTClockMetric new],
            [XCTMemoryMetric new],
          ]
                            options:XCTMeasureOptions.defaultOptions
                              block:^{
                                XCTAssertEqual(module->forward(inputs).error(),
                                               Error::Ok);
                              }];
        });
        class_addMethod([self class],
                        testForwardSelector,
                        testForwardImplementation,
                        "v@:");
      }
    }
  }
}

@end
