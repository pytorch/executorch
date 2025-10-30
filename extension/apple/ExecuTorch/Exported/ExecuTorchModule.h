/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchValue.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds the static metadata for a single tensor: its shape, layout,
 * element type, whether its memory was pre-planned by the runtime,
 * and its debug name.
 */
NS_SWIFT_NAME(TensorMetadata)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchTensorMetadata : NSObject

/** The size of each dimension. */
@property (nonatomic, readonly) NSArray<NSNumber *> *shape
    NS_REFINED_FOR_SWIFT;

/** The order in which dimensions are laid out. */
@property (nonatomic, readonly) NSArray<NSNumber *> *dimensionOrder
    NS_REFINED_FOR_SWIFT;

/** The scalar type of each element in the tensor. */
@property (nonatomic, readonly) ExecuTorchDataType dataType;

/** YES if the runtime pre-allocated memory for this tensor. */
@property (nonatomic, readonly) BOOL isMemoryPlanned;

/** The (optional) user-visible name of this tensor (may be empty) */
@property (nonatomic, readonly) NSString *name;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 * Encapsulates all of the metadata for a loaded method: its name,
 * how many inputs/outputs/attributes it has, per-argument tags,
 * per-tensor metadata, buffer sizes, backends, and instruction count.
 */
NS_SWIFT_NAME(MethodMetadata)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchMethodMetadata : NSObject

/** The method’s name. */
@property (nonatomic, readonly) NSString *name;

/** An array of ExecuTorchValueTag raw values, one per declared input. */
@property (nonatomic, readonly) NSArray<NSNumber *> *inputValueTags
    NS_REFINED_FOR_SWIFT;

/** An array of ExecuTorchValueTag raw values, one per declared output. */
@property (nonatomic, readonly) NSArray<NSNumber *> *outputValueTags
    NS_REFINED_FOR_SWIFT;

/**
 * Mapping from input-index to TensorMetadata.
 * Only present for those indices whose tag == .tensor
 */
@property (nonatomic, readonly) NSDictionary<NSNumber *, ExecuTorchTensorMetadata *> *inputTensorMetadata
    NS_REFINED_FOR_SWIFT;

/**
 * Mapping from output-index to TensorMetadata.
 * Only present for those indices whose tag == .tensor
 */
@property (nonatomic, readonly) NSDictionary<NSNumber *, ExecuTorchTensorMetadata *> *outputTensorMetadata
    NS_REFINED_FOR_SWIFT;

/** A list of attribute TensorsMetadata. */
@property (nonatomic, readonly) NSArray<ExecuTorchTensorMetadata *> *attributeTensorMetadata;

/** A list of memory-planned buffer sizes. */
@property (nonatomic, readonly) NSArray<NSNumber *> *memoryPlannedBufferSizes
    NS_REFINED_FOR_SWIFT;

/** Names of all backends this method can run on. */
@property (nonatomic, readonly) NSArray<NSString *> *backendNames;

/** Total number of low-level instructions in this method’s body. */
@property (nonatomic, readonly) NSInteger instructionCount;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/**
 * Enum to define loading behavior.
 * Values can be a subset, but must numerically match exactly those defined in
 * extension/module/module.h
 */
typedef NS_ENUM(NSInteger, ExecuTorchModuleLoadMode) {
  ExecuTorchModuleLoadModeFile = 0,
  ExecuTorchModuleLoadModeMmap,
  ExecuTorchModuleLoadModeMmapUseMlock,
  ExecuTorchModuleLoadModeMmapUseMlockIgnoreErrors,
} NS_SWIFT_NAME(ModuleLoadMode);

/**
 * Enum to define the verification level used when loading a module.
 * Values can be a subset, but must numerically match exactly those defined in
 * runtime/executor/program.h
 */
typedef NS_ENUM(uint8_t, ExecuTorchVerification) {
    ExecuTorchVerificationMinimal,
    ExecuTorchVerificationInternalConsistency,
} NS_SWIFT_NAME(ModuleVerification);

/**
 * Represents a module that encapsulates an ExecuTorch program.
 * This class is a facade for loading programs and executing methods within them.
 */
NS_SWIFT_NAME(Module)
@interface ExecuTorchModule : NSObject

/**
 * Initializes a module with a file path, data path and a specified load mode.
 *
 * @param filePath A string representing the path to the ExecuTorch program file.
 * @param dataFilePaths A list of strings representing paths to .ptd files with
 * external tensors and external data.
 * @param loadMode A value from ExecuTorchModuleLoadMode that determines the
 * file loading behavior.
 * @return An initialized ExecuTorchModule instance.
 */
- (instancetype)initWithFilePath:(NSString *)filePath
                   dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                        loadMode:(ExecuTorchModuleLoadMode)loadMode
    NS_DESIGNATED_INITIALIZER;

/**
 * Initializes a module with a file path, data path and a specified load mode.
 *
 * @param filePath A string representing the path to the ExecuTorch program file.
 * @param dataFilePaths A list of strings representing paths to .ptd files with
 * external tensors and external data.
 * @return An initialized ExecuTorchModule instance.
 */
- (instancetype)initWithFilePath:(NSString *)filePath
                   dataFilePaths:(NSArray<NSString *> *)dataFilePaths;

/**
 * Initializes a module with a file path and a specified load mode.
 *
 * @param filePath A string representing the path to the ExecuTorch program file.
 * @param loadMode A value from ExecuTorchModuleLoadMode that determines the file loading behavior.
 * @return An initialized ExecuTorchModule instance.
 */
- (instancetype)initWithFilePath:(NSString *)filePath
                        loadMode:(ExecuTorchModuleLoadMode)loadMode;

/**
 * Initializes a module with a file path using the default load mode (File mode).
 *
 * @param filePath A string representing the path to the ExecuTorch program file.
 * @return An initialized ExecuTorchModule instance.
 */
- (instancetype)initWithFilePath:(NSString *)filePath;

/**
 * Loads the module’s program using the specified verification level.
 *
 * @param verification The verification level to apply when loading the program.
 * @param error A pointer to an NSError pointer that will be set if an error occurs.
 * @return YES if the program was successfully loaded; otherwise, NO.
 */
- (BOOL)loadWithVerification:(ExecuTorchVerification)verification
                       error:(NSError **)error;

/**
 * Loads the module’s program using minimal verification.
 *
 * This is a convenience overload that defaults the verification level to Minimal.
 *
 * @param error A pointer to an NSError pointer that will be set if an error occurs.
 * @return YES if the program was successfully loaded; otherwise, NO.
 */
- (BOOL)load:(NSError **)error;

/**
 * Checks if the module is loaded.
 *
 * @return YES if the module's program is loaded; otherwise, NO.
 */
- (BOOL)isLoaded;

/**
 * Loads a specific method from the program.
 *
 * @param methodName A string representing the name of the method to load.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return YES if the method was successfully loaded; otherwise, NO.
 */
- (BOOL)loadMethod:(NSString *)methodName
             error:(NSError **)error NS_SWIFT_NAME(load(_:));

/**
 * Checks if a specific method is loaded.
 *
 * @param methodName A string representing the method name.
 * @return YES if the method is loaded; otherwise, NO.
 */
- (BOOL)isMethodLoaded:(NSString *)methodName NS_SWIFT_NAME(isLoaded(_:));

/**
 * Unloads a method and releases its native resources and planned buffers.
 *
 * @param methodName The method to unload.
 * @return YES if the method was unloaded; NO if it was not loaded at all.
 */
- (BOOL)unloadMethod:(NSString *)methodName NS_SWIFT_NAME(unload(_:));

/**
 * Retrieves the set of method names available in the loaded program.
 *
 * The method names are returned as an unordered set of strings. The program and methods
 * are loaded as needed.
 *
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An unordered set of method names, or nil in case of an error.
 */
- (nullable NSSet<NSString *> *)methodNames:(NSError **)error
    NS_RETURNS_RETAINED;

/**
 * Retrieves full metadata for a particular method in the loaded module.
 *
 * This includes the method’s name, input/output value tags, tensor shapes
 * and layouts, buffer sizes, backend support list, and instruction count.
 *
 * @param methodName A string representing the method name.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An ExecuTorchMethodMetadata object on success, or nil if the method isn’t found or a load error occurred.
 */
 - (nullable ExecuTorchMethodMetadata *)methodMetadata:(NSString *)methodName
                                                 error:(NSError **)error
    NS_RETURNS_RETAINED;

/**
 * Executes a specific method with the provided input values.
 *
 * The method is loaded on demand if not already loaded.
 *
 * @param methodName A string representing the method name.
 * @param values An NSArray of ExecuTorchValue objects representing the inputs.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                            withInputs:(NSArray<ExecuTorchValue *> *)values
                                                 error:(NSError **)error
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Executes a specific method with the provided single input value.
 *
 * The method is loaded on demand if not already loaded.
 *
 * @param methodName A string representing the method name.
 * @param value An ExecuTorchValue object representing the input.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                             withInput:(ExecuTorchValue *)value
                                                 error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes a specific method with no input values.
 *
 * The method is loaded on demand if not already loaded.
 *
 * @param methodName A string representing the method name.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                                 error:(NSError **)error
    NS_SWIFT_NAME(execute(_:))
    NS_RETURNS_RETAINED;

/**
 * Executes a specific method with the provided input tensors.
 *
 * The method is loaded on demand if not already loaded.
 *
 * @param methodName A string representing the method name.
 * @param tensors An NSArray of ExecuTorchTensor objects representing the inputs.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                           withTensors:(NSArray<ExecuTorchTensor *> *)tensors
                                                 error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes a specific method with the provided single input tensor.
 *
 * The method is loaded on demand if not already loaded.
 *
 * @param methodName A string representing the method name.
 * @param tensor An ExecuTorchTensor object representing the input.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                            withTensor:(ExecuTorchTensor *)tensor
                                                 error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes the "forward" method with the provided input values.
 *
 * This is a convenience method that calls the executeMethod with "forward" as the method name.
 *
 * @param values An NSArray of ExecuTorchValue objects representing the inputs.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)forwardWithInputs:(NSArray<ExecuTorchValue *> *)values
                                                     error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes the "forward" method with the provided single input value.
 *
 * This is a convenience method that calls the executeMethod with "forward" as the method name.
 *
 * @param value An ExecuTorchValue object representing the input.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)forwardWithInput:(ExecuTorchValue *)value
                                                    error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes the "forward" method with no inputs.
 *
 * This is a convenience method that calls the executeMethod with "forward" as the method name.
 *
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)forward:(NSError **)error
    NS_RETURNS_RETAINED;

/**
 * Executes the "forward" method with the provided input tensors.
 *
 * This is a convenience method that calls the executeMethod with "forward" as the method name.
 *
 * @param tensors An NSArray of ExecuTorchTensor objects representing the inputs.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)forwardWithTensors:(NSArray<ExecuTorchTensor *> *)tensors
                                                      error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Executes the "forward" method with the provided single input tensor.
 *
 * This is a convenience method that calls the executeMethod with "forward" as the method name.
 *
 * @param tensor An ExecuTorchTensor object representing the input.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return An NSArray of ExecuTorchValue objects representing the outputs, or nil in case of an error.
 */
- (nullable NSArray<ExecuTorchValue *> *)forwardWithTensor:(ExecuTorchTensor *)tensor
                                                     error:(NSError **)error
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Sets a single input value for the "forward" method at index 0.
 *
 * @param value The input value.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInput:(ExecuTorchValue *)value
           error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets a single input value for the "forward" method at the specified index.
 *
 * @param value The input value.
 * @param index Zero-based input index.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInput:(ExecuTorchValue *)value
         atIndex:(NSInteger)index
           error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets a single input value for the specified method at index 0.
 *
 * @param value The input value.
 * @param methodName The method name.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInput:(ExecuTorchValue *)value
       forMethod:(NSString *)methodName
           error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets a single input value for the specified method at the given index.
 *
 * The module retains the provided value to keep its backing storage alive
 * until the value is overwritten or the module is deallocated.
 *
 * @param value The input value.
 * @param methodName The method name.
 * @param index Zero-based input index.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInput:(ExecuTorchValue *)value
       forMethod:(NSString *)methodName
         atIndex:(NSInteger)index
           error:(NSError **)error NS_REFINED_FOR_SWIFT;

/**
 * Sets all input values for the "forward" method.
 *
 * The number and types of values must match the method’s declared inputs.
 *
 * @param values The input values, one per declared input.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInputs:(NSArray<ExecuTorchValue *> *)values
            error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets all input values for the specified method.
 *
 * The module retains the provided values to keep their backing storage alive
 * until the values are overwritten or the module is deallocated.
 *
 * @param values The input values, one per declared input.
 * @param methodName The method name.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setInputs:(NSArray<ExecuTorchValue *> *)values
        forMethod:(NSString *)methodName
            error:(NSError **)error NS_REFINED_FOR_SWIFT;

/**
 * Sets the output buffer for the "forward" method at index 0.
 *
 * Only tensor outputs are supported. The provided value must wrap a tensor
 * compatible with the method’s output slot.
 *
 * @param value The output buffer (must wrap a tensor).
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setOutput:(ExecuTorchValue *)value
            error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets the output buffer for the "forward" method at the specified index.
 *
 * Only tensor outputs are supported. The provided value must wrap a tensor
 * compatible with the method’s output slot.
 *
 * @param value The output buffer (must wrap a tensor).
 * @param index Zero-based output index.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setOutput:(ExecuTorchValue *)value
          atIndex:(NSInteger)index
            error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets the output buffer for the specified method at index 0.
 *
 * Only tensor outputs are supported. The provided value must wrap a tensor
 * compatible with the method’s output slot.
 *
 * @param value The output buffer (must wrap a tensor).
 * @param methodName The method name.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setOutput:(ExecuTorchValue *)value
        forMethod:(NSString *)methodName
            error:(NSError **)error NS_SWIFT_UNAVAILABLE("");

/**
 * Sets the output buffer for the specified method at the given index.
 *
 * The module retains the provided value to keep its backing storage alive
 * until the value is overwritten or the module is deallocated.
 * Only tensor outputs are supported.
 *
 * @param value The output buffer (must wrap a tensor).
 * @param methodName The method name.
 * @param index Zero-based output index.
 * @param error On failure, set to an NSError describing the issue.
 * @return YES on success; NO otherwise.
 */
- (BOOL)setOutput:(ExecuTorchValue *)value
        forMethod:(NSString *)methodName
          atIndex:(NSInteger)index
            error:(NSError **)error NS_REFINED_FOR_SWIFT;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
