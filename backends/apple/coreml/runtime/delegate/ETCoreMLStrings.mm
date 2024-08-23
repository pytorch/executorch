//
// ETCoreMLStrings.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import "ETCoreMLStrings.h"

@implementation ETCoreMLStrings

+ (NSString *)productName {
    static NSString * const ETCoreMLProductName = @"executorchcoreml";
    return ETCoreMLProductName;
}

+ (NSString *)productIdentifier {
    static NSString * const ETCoreMLProductIdentifier = @"com.apple.executorchcoreml";
    return ETCoreMLProductIdentifier;
}

+ (NSString *)delegateIdentifier {
    static NSString * const ETCoreMLDelegateIdentifier = @"CoreMLBackend";
    return ETCoreMLDelegateIdentifier;
}

+ (NSString *)configPlistName {
    static NSString * const ETCoreMLDelegateConfigPlistName = @"com.apple.executorchcoreml_config";
    return ETCoreMLDelegateConfigPlistName;
}

+ (NSString *)computeUnitsKeyName {
    static NSString * const ETCoreMLComputeUnitsName = @"compute_units";
    return ETCoreMLComputeUnitsName;
}

+ (NSString *)compiledModelExtensionName {
    static NSString * const ETCoreMLCompiledModelExtensionName = @"mlmodelc";
    return ETCoreMLCompiledModelExtensionName;
}

+ (NSString *)modelExtensionName {
    static NSString * const ETCoreMLModelExtensionName = @"mlpackage";
    return ETCoreMLModelExtensionName;
}

+ (NSString *)metadataFileRelativePath {
    static NSString * const ETCoreMLMetadataFileRelativePath = @"metadata.json";
    return ETCoreMLMetadataFileRelativePath;
}

+ (NSString *)modelFileRelativePath {
    static NSString * const ETCoreMLModelFileRelativePath = @"model.mlpackage";
    return ETCoreMLModelFileRelativePath;
}

+ (NSString *)compiledModelFileRelativePath {
    static NSString * const ETCoreMLCompiledModelFileRelativePath = @"model.mlmodelc";
    return ETCoreMLCompiledModelFileRelativePath;
}

+ (NSString *)cpuComputeUnitName {
    static NSString * const ETCoreMLCPUComputeUnitName = @"cpu_only";
    return ETCoreMLCPUComputeUnitName;
}

+ (NSString *)cpuAndGpuComputeUnitsName {
    static NSString * const ETCoreMLCPUAndGPUComputeUnitsName = @"cpu_and_gpu";
    return ETCoreMLCPUAndGPUComputeUnitsName;
}

+ (NSString *)cpuAndNeuralEngineComputeUnitsName {
    static NSString * const ETCoreMLCPUAndNeuralEngineComputeUnitsName = @"cpu_and_ne";
    return ETCoreMLCPUAndNeuralEngineComputeUnitsName;
}

+ (NSString *)allComputeUnitsName {
    static NSString * const ETCoreMLAllComputeUnitsName = @"all";
    return ETCoreMLAllComputeUnitsName;
}

+ (NSString *)databaseName {
    static NSString * const ETCoreMLDatabaseName = @"assets.db";
    return ETCoreMLDatabaseName;
}

+ (NSString *)debugInfoFileRelativePath {
    static NSString * const ETCoreMLDebugInfoFileRelativePath = @"debug_info.json";
    return ETCoreMLDebugInfoFileRelativePath;
}

+ (NSString *)debugSymbolToOperationPathKeyName {
    static NSString * const ETCoreMLDebugSymbolToOperationPathKeyName = @"debugSymbolToOperationPath";
    return ETCoreMLDebugSymbolToOperationPathKeyName;
}

+ (NSString *)debugSymbolToHandlesKeyName {
    static NSString * const ETCoreMLDebugSymbolToHandlesKeyName = @"debugSymbolToHandles";
    return ETCoreMLDebugSymbolToHandlesKeyName;
}

+ (nullable NSString *)assetsDirectoryPath {
    static dispatch_once_t onceToken;
    static NSString *result = nil;
    dispatch_once(&onceToken, ^{
        NSArray<NSString *> *paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
        if (paths.count > 0) {
            result = [paths.lastObject stringByAppendingPathComponent:self.productName];
        }
    });
    
    return result;
}

+ (nullable NSString *)trashDirectoryPath {
    static dispatch_once_t onceToken;
    static NSString *result = nil;
    dispatch_once(&onceToken, ^{
        result = [NSTemporaryDirectory() stringByAppendingPathComponent:self.productName];
    });
    
    return result;
}

+ (nullable NSString *)databaseDirectoryPath {
    static dispatch_once_t onceToken;
    static NSString *result = nil;
    dispatch_once(&onceToken, ^{
        NSArray<NSString *> *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
        if (paths.count > 0) {
            result = [paths.lastObject stringByAppendingPathComponent:self.productName];
        }
    });
    
    return result;
}


@end
