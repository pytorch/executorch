/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ExecuTorchTextLLMRunnerErrorDomain;

/**
 A wrapper class for the C++ llm::TextLLMRunner that provides
 Objective-C APIs to load models, manage tokenization with custom
 special tokens, generate text sequences, and stop the runner.
*/
NS_SWIFT_NAME(TextLLMRunner)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchTextLLMRunner : NSObject

/**
 Initializes a text LLM runner with the given model and tokenizer paths,
 and a list of special tokens to include in the tokenizer.

 @param modelPath      File system path to the serialized model.
 @param tokenizerPath  File system path to the tokenizer data.
 @param tokens         An array of NSString special tokens to use during tokenization.
 @return An initialized ExecuTorchTextLLMRunner instance.
*/
- (instancetype)initWithModelPath:(NSString *)modelPath
                    tokenizerPath:(NSString *)tokenizerPath
                    specialTokens:(NSArray<NSString *> *)tokens;

/**
 Checks whether the underlying model has been successfully loaded.

 @return YES if the model is loaded, NO otherwise.
*/
- (BOOL)isLoaded;

/**
 Loads the model into memory, returning an error if loading fails.

 @param error   On failure, populated with an NSError explaining the issue.
 @return YES if loading succeeds, NO if an error occurred.
*/
- (BOOL)loadWithError:(NSError **)error;

/**
 Generates text given an input prompt, up to a specified sequence length.
 Invokes the provided callback for each generated token.

 @param prompt    The initial text prompt to generate from.
 @param seq_len   The maximum number of tokens to generate.
 @param callback  A block called with each generated token as an NSString.
 @param error     On failure, populated with an NSError explaining the issue.
 @return YES if generation completes successfully, NO if an error occurred.
*/
- (BOOL)generate:(NSString *)prompt
   sequenceLength:(NSInteger)seq_len
withTokenCallback:(nullable void (^)(NSString *))callback
            error:(NSError **)error;

/**
 Stops any ongoing generation and cleans up internal resources.
*/
- (void)stop;

@end

NS_ASSUME_NONNULL_END
