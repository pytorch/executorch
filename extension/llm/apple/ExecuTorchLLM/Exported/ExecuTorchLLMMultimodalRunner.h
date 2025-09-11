/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 Types of multimodal inputs supported by the ExecuTorch LLM APIs.
 Must be in sync with the C++ enum in llm/runner/multimodal_input.h
*/
typedef NS_ENUM(NSInteger, ExecuTorchLLMMultimodalInputType) {
  ExecuTorchLLMMultimodalInputTypeText,
  ExecuTorchLLMMultimodalInputTypeImage,
  ExecuTorchLLMMultimodalInputTypeAudio,
  ExecuTorchLLMMultimodalInputTypeUnsupported,
} NS_SWIFT_NAME(MultimodalInputType);

/**
 A container for image inputs used with multimodal generation APIs.
*/
NS_SWIFT_NAME(Image)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchLLMImage : NSObject<NSCopying>

/**
 Initializes an image container with the provided data and dimensions.

 @param data       Raw image bytes.
 @param width      Image width in pixels.
 @param height     Image height in pixels.
 @param channels   Number of channels.
 @return An initialized ExecuTorchLLMImage instance.
*/
- (instancetype)initWithData:(NSData *)data
                       width:(NSInteger)width
                      height:(NSInteger)height
                    channels:(NSInteger)channels
    NS_DESIGNATED_INITIALIZER;

@property(nonatomic, readonly) NSData *data;
@property(nonatomic, readonly) NSInteger width;
@property(nonatomic, readonly) NSInteger height;
@property(nonatomic, readonly) NSInteger channels;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 A container for pre-processed audio features.
*/
NS_SWIFT_NAME(Audio)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchLLMAudio : NSObject<NSCopying>

/**
 Initializes an audio features container with the provided data and shape.

 @param data        Feature buffer.
 @param batchSize   Batch dimension size.
 @param bins        Number of frequency bins.
 @param frames      Number of time frames.
 @return An initialized ExecuTorchLLMAudio instance.
*/
- (instancetype)initWithData:(NSData *)data
                   batchSize:(NSInteger)batchSize
                        bins:(NSInteger)bins
                      frames:(NSInteger)frames
    NS_DESIGNATED_INITIALIZER;

@property(nonatomic, readonly) NSData *data;
@property(nonatomic, readonly) NSInteger batchSize;
@property(nonatomic, readonly) NSInteger bins;
@property(nonatomic, readonly) NSInteger frames;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 A tagged container for a single multimodal input item used by
 multimodal generation APIs.
*/
NS_SWIFT_NAME(MultimodalInput)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchLLMMultimodalInput : NSObject<NSCopying>

/**
 Creates a text input.

 @param text  The UTF-8 text to provide as input.
 @return A retained ExecuTorchLLMMultimodalInput instance of type Text.
*/
+ (instancetype)inputWithText:(NSString *)text
    NS_SWIFT_NAME(init(_:))
    NS_RETURNS_RETAINED;

/**
 Creates an image input.

 @param image  The image payload to provide as input.
 @return A retained ExecuTorchLLMMultimodalInput instance of type Image.
*/
+ (instancetype)inputWithImage:(ExecuTorchLLMImage *)image
    NS_SWIFT_NAME(init(_:))
    NS_RETURNS_RETAINED;

/**
 Creates an audio-features input.

 @param audio  The pre-processed audio features to provide as input.
 @return A retained ExecuTorchLLMMultimodalInput instance of type Audio.
*/
+ (instancetype)inputWithAudio:(ExecuTorchLLMAudio *)audio
    NS_SWIFT_NAME(init(audio:))
    NS_RETURNS_RETAINED;

@property(nonatomic, readonly) ExecuTorchLLMMultimodalInputType type;
@property(nonatomic, readonly, nullable) NSString *text;
@property(nonatomic, readonly, nullable) ExecuTorchLLMImage *image;
@property(nonatomic, readonly, nullable) ExecuTorchLLMAudio *audio;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 A wrapper class for the C++ llm::MultimodalLLMRunner that provides
 Objective-C APIs to load models, manage tokenization, accept mixed
 input modalities, generate text sequences, and stop the runner.
*/
NS_SWIFT_NAME(MultimodalRunner)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchLLMMultimodalRunner : NSObject

/**
 Initializes a multimodal LLM runner with the given model and tokenizer paths.

 @param modelPath      File system path to the serialized model.
 @param tokenizerPath  File system path to the tokenizer data.
 @return An initialized ExecuTorchLLMMultimodalRunner instance.
*/
- (instancetype)initWithModelPath:(NSString *)modelPath
                    tokenizerPath:(NSString *)tokenizerPath
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
 Generates text given a list of multimodal inputs, up to a specified sequence length.
 Invokes the provided callback for each generated token.

 @param inputs    An ordered array of multimodal inputs.
 @param seq_len   The maximum number of tokens to generate.
 @param callback  A block called with each generated token as an NSString.
 @param error     On failure, populated with an NSError explaining the issue.
 @return YES if generation completes successfully, NO if an error occurred.
*/
- (BOOL)generate:(NSArray<ExecuTorchLLMMultimodalInput *> *)inputs
   sequenceLength:(NSInteger)seq_len
withTokenCallback:(nullable void (^)(NSString *))callback
            error:(NSError **)error;

/**
 Stops any ongoing generation and cleans up internal resources.
*/
- (void)stop;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
