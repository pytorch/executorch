/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLLMConfig.h"

NS_ASSUME_NONNULL_BEGIN

/**
 A wrapper class for the C++ llm::TextLLMRunner that provides
 Objective-C APIs to load models, manage tokenization with custom
 special tokens, generate text sequences, and stop the runner.
*/
NS_SWIFT_NAME(TextRunner)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchLLMTextRunner : NSObject

/**
 Initializes a text LLM runner with the given model and tokenizer paths,
 and a list of special tokens to include in the tokenizer.

 @param modelPath      File system path to the serialized model.
 @param tokenizerPath  File system path to the tokenizer data.
 @return An initialized ExecuTorchLLMTextRunner instance.
*/
- (instancetype)initWithModelPath:(NSString *)modelPath
                    tokenizerPath:(NSString *)tokenizerPath;

/**
 Initializes a text LLM runner with the given model and tokenizer paths,
 and a list of special tokens to include in the tokenizer.

 @param modelPath      File system path to the serialized model.
 @param tokenizerPath  File system path to the tokenizer data.
 @param specialTokens  An array of NSString special tokens to use during tokenization.
 @return An initialized ExecuTorchLLMTextRunner instance.
*/
- (instancetype)initWithModelPath:(NSString *)modelPath
                    tokenizerPath:(NSString *)tokenizerPath
                    specialTokens:(NSArray<NSString *> *)specialTokens
    NS_DESIGNATED_INITIALIZER;

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
 Generates text given an input prompt. A default configuration
 is created and passed to the configuration block for in-place mutation.

 The token callback, if provided, is invoked for each generated token.

 @param prompt     The initial text prompt to generate from.
 @param config     A configuration object.
 @param callback   A block called with each generated token as an NSString.
 @param error      On failure, populated with an NSError explaining the issue.
 @return YES if generation completes successfully, NO if an error occurred.
*/
- (BOOL)generateWithPrompt:(NSString *)prompt
                    config:(ExecuTorchLLMConfig *)config
             tokenCallback:(nullable void (^)(NSString *token))callback
                     error:(NSError **)error
    NS_SWIFT_NAME(generate(_:_:tokenCallback:));

/**
 Stop producing new tokens and terminate the current generation process.
*/
- (void)stop;

/**
 Remove the prefilled tokens from the KV cache and reset the start position
 to 0. It also clears the stats for previous runs.
*/
- (void)reset;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
