//
// ETCoreMLStrings.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// A class to query all the constants in the system.
@interface ETCoreMLStrings : NSObject

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/// The product name.
@property (class, copy, readonly, nonatomic) NSString* productName;
/// The product identifier.
@property (class, copy, readonly, nonatomic) NSString* productIdentifier;

/// The identifier that's used to register the delegate.
@property (class, copy, readonly, nonatomic) NSString* delegateIdentifier;

/// The delegate plist config name.
@property (class, copy, readonly, nonatomic) NSString* configPlistName;

/// The key name for compute units.
@property (class, copy, readonly, nonatomic) NSString* computeUnitsKeyName;

/// The compiled model package extension name.
@property (class, copy, readonly, nonatomic) NSString* compiledModelExtensionName;

/// The model package extension name.
@property (class, copy, readonly, nonatomic) NSString* modelExtensionName;

/// The model metadata relative path in the AOT blob.
@property (class, copy, readonly, nonatomic) NSString* metadataFileRelativePath;
/// The model package relative path in the AOT blob.
@property (class, copy, readonly, nonatomic) NSString* modelFileRelativePath;
/// The compiled model relative path in the AOT blob.
@property (class, copy, readonly, nonatomic) NSString* compiledModelFileRelativePath;

/// The default assets directory path.
@property (class, copy, readonly, nonatomic, nullable) NSString* assetsDirectoryPath;
/// The default trash directory path.
@property (class, copy, readonly, nonatomic, nullable) NSString* trashDirectoryPath;

/// The default database directory path.
@property (class, copy, readonly, nonatomic, nullable) NSString* databaseDirectoryPath;
/// The default database name.
@property (class, copy, readonly, nonatomic) NSString* databaseName;

/// CPU compute unit name.
@property (class, copy, readonly, nonatomic) NSString* cpuComputeUnitName;
/// GPU compute unit name.
@property (class, copy, readonly, nonatomic) NSString* cpuAndGpuComputeUnitsName;
/// NeuralEngine compute unit name.
@property (class, copy, readonly, nonatomic) NSString* cpuAndNeuralEngineComputeUnitsName;
/// All compute units name.
@property (class, copy, readonly, nonatomic) NSString* allComputeUnitsName;

/// The debug info relative path in the AOT blob.
@property (class, copy, readonly, nonatomic, nullable) NSString* debugInfoFileRelativePath;
/// The debug symbol to operation path key name.
@property (class, copy, readonly, nonatomic, nullable) NSString* debugSymbolToOperationPathKeyName;
/// The debug symbol to handles key name.
@property (class, copy, readonly, nonatomic, nullable) NSString* debugSymbolToHandlesKeyName;

@end

NS_ASSUME_NONNULL_END
