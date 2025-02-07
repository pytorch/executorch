//
// ETCoreMLTestUtils.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"

#import "ETCoreMLAsset.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModelDebugInfo.h"
#import "ETCoreMLModelAnalyzer.h"
#import "ETCoreMLModelCompiler.h"
#import "ETCoreMLStrings.h"
#import <filesystem>
#import "inmemory_filesystem_utils.hpp"
#import <iostream>
#import <memory>
#import "model_metadata.h"

namespace {
NSURL * _Nullable create_directory_if_needed(NSURL *url,
                                             NSFileManager *fm,
                                             NSError * __autoreleasing *error) {
    if (![fm fileExistsAtPath:url.path] && ![fm createDirectoryAtURL:url withIntermediateDirectories:YES attributes:@{} error:error]) {
        return nil;
    }
    
    return url;
}

ETCoreMLAssetManager * _Nullable create_asset_manager(NSString *assets_directory_path,
                                                      NSString *trash_directory_path,
                                                      NSString *database_directory_path,
                                                      NSString *database_name,
                                                      NSError * __autoreleasing *error) {
    NSFileManager *fm  = [[NSFileManager alloc] init];
    
    NSURL *assets_directory_url = [NSURL fileURLWithPath:assets_directory_path];
    if (!create_directory_if_needed(assets_directory_url, fm, error)) {
        return nil;
    }
    
    NSURL *trash_directory_url = [NSURL fileURLWithPath:trash_directory_path];
    if (!create_directory_if_needed(trash_directory_url, fm, error)) {
        return nil;
    }
    
    NSURL *database_directory_url = [NSURL fileURLWithPath:database_directory_path];
    if (!create_directory_if_needed(database_directory_url, fm, error)) {
        return nil;
    }
    
    NSURL *database_url = [database_directory_url URLByAppendingPathComponent:database_name];
    ETCoreMLAssetManager *manager = [[ETCoreMLAssetManager alloc] initWithDatabaseURL:database_url
                                                                   assetsDirectoryURL:assets_directory_url
                                                                    trashDirectoryURL:trash_directory_url
                                                                 maxAssetsSizeInBytes:2 * 1024 * 1024 // 2 mb
                                                                                error:error];
    return manager;
}

template<typename T>
T toValue(NSNumber *value);

template<>
int32_t toValue(NSNumber *value) {
    return value.intValue;
}

template<>
double toValue(NSNumber *value) {
    return value.doubleValue;
}

template<>
float toValue(NSNumber *value) {
    return value.floatValue;
}

template<>
_Float16 toValue(NSNumber *value) {
    return static_cast<_Float16>(value.floatValue);
}

template <typename T>
void fillBytesWithValue(void *mutableBytes, NSInteger size, NSNumber *value) {
    T *start = reinterpret_cast<T *>(mutableBytes);
    T *end = start + size;
    T fillValue = toValue<T>(value);
    std::fill(start, end, fillValue);
}

bool extract_model_metadata(const inmemoryfs::InMemoryFileSystem& inMemoryFS,
                            executorchcoreml::ModelMetadata& model_metadata) {
    std::error_code ec;
    auto buffer = inMemoryFS.get_file_content({ETCoreMLStrings.metadataFileRelativePath.UTF8String}, ec);
    if (!buffer) {
        return false;
    }
    
    std::string contents;
    contents.assign(reinterpret_cast<char *>(buffer->data()), buffer->size());
    model_metadata.from_json_string(std::move(contents));
    return true;
}

ETCoreMLAsset * _Nullable make_asset(NSURL *url,
                                     NSString *identifier,
                                     NSFileManager *fm,
                                     NSError * __autoreleasing *error) {
    auto backingAsset = executorchcoreml::Asset::make(url, identifier, fm, error);
    if (!backingAsset) {
        return nil;
    }
    
    return [[ETCoreMLAsset alloc] initWithBackingAsset:std::move(backingAsset.value())];
}

}

@implementation ETCoreMLTestUtils

+ (nullable NSURL *)createUniqueAssetInDirectoryAtURL:(NSURL *)directoryURL
                                          withContent:(NSString *)content
                                          fileManager:(NSFileManager *)fileManager
                                                error:(NSError * __autoreleasing *)error {
    NSURL *assetURL = [directoryURL URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    if (![fileManager createDirectoryAtURL:assetURL withIntermediateDirectories:NO attributes:@{} error:error]) {
        return nil;
    }
    
    NSURL *dirURL = [assetURL URLByAppendingPathComponent:@"folder"];
    if (![fileManager createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:error]) {
        return nil;
    }
    
    NSURL *fileURL = [dirURL URLByAppendingPathComponent:@"content.txt"];
    NSData* data = [content dataUsingEncoding:NSUTF8StringEncoding];
    if (![data writeToURL:fileURL options:NSDataWritingAtomic error:error]) {
        return nil;
    }
    
    return assetURL;
}

+ (nullable ETCoreMLAssetManager *)createAssetManagerWithURL:(NSURL *)directoryURL
                                                       error:(NSError * __autoreleasing *)error {
    NSString *assetDirectoryPath = [directoryURL.path stringByAppendingPathComponent:@"assets"];
    NSString *trashDirectoryPath = [directoryURL.path stringByAppendingPathComponent:@"trash"];
    NSString *databaseDirectoryPath = directoryURL.path;
    return create_asset_manager(assetDirectoryPath, trashDirectoryPath, databaseDirectoryPath, @"main.db", error);
}

+ (nullable MLMultiArray *)filledMultiArrayWithShape:(NSArray<NSNumber *> *)shape
                                            dataType:(MLMultiArrayDataType)dataType
                                       repeatedValue:(NSNumber *)value
                                               error:(NSError * __autoreleasing *)error {
    MLMultiArray *multiArray = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:error];
    if (!multiArray) {
        return nil;
    }
    
    [multiArray getMutableBytesWithHandler:^(void *mutableBytes, NSInteger __unused size, NSArray<NSNumber *> * __unused strides) {
        switch (multiArray.dataType) {
            case MLMultiArrayDataTypeFloat16: {
                fillBytesWithValue<_Float16>(mutableBytes, multiArray.count, value);
                break;
            }
            case MLMultiArrayDataTypeFloat: {
                fillBytesWithValue<float>(mutableBytes, multiArray.count, value);
                break;
            }
            case MLMultiArrayDataTypeDouble: {
                fillBytesWithValue<double>(mutableBytes, multiArray.count, value);
                break;
            }
                
            case MLMultiArrayDataTypeInt32: {
                fillBytesWithValue<int>(mutableBytes, multiArray.count, value);
                break;
            }
                
            default:
                break;
        }
    }];
    
    return multiArray;
    
}

+ (nullable MLMultiArray *)filledMultiArrayWithConstraint:(MLMultiArrayConstraint *)constraint
                                            repeatedValue:(NSNumber *)value
                                                    error:(NSError * __autoreleasing *)error {
    return [self filledMultiArrayWithShape:constraint.shape dataType:constraint.dataType repeatedValue:value error:error];
}

+ (nullable NSArray<MLMultiArray *> *)inputsForModel:(ETCoreMLModel *)model
                                      repeatedValues:(NSArray<NSNumber *> *)repeatedValues
                                               error:(NSError * __autoreleasing *)error {
    NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptionsByName = [model.mlModel.modelDescription inputDescriptionsByName];
    NSMutableArray<MLMultiArray *> *inputs = [NSMutableArray arrayWithCapacity:inputDescriptionsByName.count];
    NSEnumerator<NSNumber *> *enumerator = [repeatedValues objectEnumerator];
    for (NSString *inputName in model.orderedInputNames) {
        MLMultiArrayConstraint *constraint = inputDescriptionsByName[inputName].multiArrayConstraint;
        MLMultiArray *multiArray = [ETCoreMLTestUtils filledMultiArrayWithConstraint:constraint repeatedValue:[enumerator nextObject] error:error];
        if (!multiArray) {
            return nil;
        }
        
        [inputs addObject:multiArray];
    }
    
    return inputs;
}

+ (nullable id<MLFeatureProvider>)inputFeaturesForModel:(ETCoreMLModel *)model
                                         repeatedValues:(NSArray<NSNumber *> *)repeatedValues
                                                  error:(NSError * __autoreleasing *)error {
    NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptionsByName = [model.mlModel.modelDescription inputDescriptionsByName];
    NSMutableDictionary<NSString *, MLFeatureValue *> *inputs = [NSMutableDictionary dictionaryWithCapacity:inputDescriptionsByName.count];
    NSEnumerator<NSNumber *> *enumerator = [repeatedValues objectEnumerator];
    for (NSString *inputName in model.orderedInputNames) {
        MLMultiArrayConstraint *constraint = inputDescriptionsByName[inputName].multiArrayConstraint;
        MLMultiArray *multiArray = [ETCoreMLTestUtils filledMultiArrayWithConstraint:constraint repeatedValue:[enumerator nextObject] error:error];
        if (!multiArray) {
            return nil;
        }
        
        MLFeatureValue *feature = [MLFeatureValue featureValueWithMultiArray:multiArray];
        inputs[inputName] = feature;
    }
    
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs
                                                             error:error];
}

+ (BOOL)extractModelAssetAndMetadataFromAOTData:(NSData *)data
                                     modelAsset:(ETCoreMLAsset *_Nullable __autoreleasing *_Nonnull)modelAsset
                                 modelDebugInfo:(ETCoreMLModelDebugInfo *_Nullable __autoreleasing *_Nonnull)modelDebugInfo
                                       metadata:(executorchcoreml::ModelMetadata&)metadata
                                         dstURL:(NSURL *)dstURL
                                    fileManager:(NSFileManager *)fileManager
                                          error:(NSError * __autoreleasing *)error {
    std::shared_ptr<inmemoryfs::MemoryBuffer> buffer = inmemoryfs::MemoryBuffer::make_unowned(const_cast<void *>(data.bytes), data.length);
    std::unique_ptr<inmemoryfs::InMemoryFileSystem> inMemoryFS = inmemoryfs::make_from_buffer(buffer);
    if (!inMemoryFS) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "%@: Model data is corrupted.",
                                      NSStringFromClass(ETCoreMLTestUtils.class));
        return NO;
    }
    
    if (!extract_model_metadata(*inMemoryFS, metadata) || !metadata.is_valid()) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedMetadata,
                                      "%@: Model metadata is corrupted.",
                                      NSStringFromClass(ETCoreMLTestUtils.class));
        return NO;
    }
    
    NSString *modelName = [NSString stringWithFormat:@"model_%s", metadata.identifier.c_str()];
    NSURL *modelURL = [dstURL URLByAppendingPathComponent:modelName];
    [fileManager removeItemAtURL:modelURL error:nil];
    if (![fileManager createDirectoryAtURL:modelURL withIntermediateDirectories:NO attributes:@{} error:error]) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorModelSaveFailed,
                                      "%@: Failed to create directory when saving model with name = %@.",
                                      NSStringFromClass(ETCoreMLTestUtils.class),
                                      modelURL.lastPathComponent);
        return NO;
    }
    
    std::filesystem::path modelPath(modelURL.fileSystemRepresentation);
    std::error_code ec;
    if (!inMemoryFS->write_item_to_disk({ETCoreMLStrings.modelFileRelativePath.UTF8String}, modelPath, true, ec)) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorModelSaveFailed,
                                      "%@: Failed to write model files to disk when saving model with name = %@.",
                                      NSStringFromClass(ETCoreMLTestUtils.class),
                                      modelURL.lastPathComponent);
        return NO;
    }
    
    ETCoreMLAsset *localAsset = ::make_asset([modelURL URLByAppendingPathComponent:ETCoreMLStrings.modelFileRelativePath],
                                             @(metadata.identifier.c_str()),
                                             fileManager,
                                             error);
    if (!localAsset) {
        return NO;
    }
    
    if (modelAsset) {
        *modelAsset = localAsset;
    }

    __block auto debugInfoBuffer = inMemoryFS->get_file_content({ETCoreMLStrings.debugInfoFileRelativePath.UTF8String}, ec);
    if (debugInfoBuffer && debugInfoBuffer->size() > 0) {

        NSData *data = [[NSData alloc] initWithBytesNoCopy:debugInfoBuffer->data()
                                                    length:debugInfoBuffer->size()
                                               deallocator:^(void * _Nonnull __unused bytes, NSUInteger __unused length) {
            debugInfoBuffer.reset();
        }];

        ETCoreMLModelDebugInfo *lModelDebugInfo = [ETCoreMLModelDebugInfo modelDebugInfoFromData:data error:nil];
        if (modelDebugInfo) {
            *modelDebugInfo = lModelDebugInfo;
        }
    }

    return YES;
}

+ (ETCoreMLModelAnalyzer *)createAnalyzerWithAOTData:(NSData *)data
                                              dstURL:(NSURL *)dstURL
                                               error:(NSError * __autoreleasing *)error {
    ETCoreMLAsset *modelAsset = nil;
    ETCoreMLModelDebugInfo *modelDebugInfo = nil;
    executorchcoreml::ModelMetadata metadata;
    NSFileManager *fileManager = [[NSFileManager alloc] init];
    if (![self extractModelAssetAndMetadataFromAOTData:data
                                            modelAsset:&modelAsset
                                        modelDebugInfo:&modelDebugInfo
                                              metadata:metadata
                                                dstURL:dstURL
                                           fileManager:fileManager
                                                 error:error]) {
        return nil;
    }
    
    NSURL *tmpCompiledModelURL = [ETCoreMLModelCompiler compileModelAtURL:modelAsset.contentURL
                                                     maxWaitTimeInSeconds:(5 * 60)
                                                                    error:error];
    if (!tmpCompiledModelURL) {
        return nil;
    }
    
    NSString *modelDirectoryName = [NSString stringWithFormat:@"model_%s", metadata.identifier.c_str()];
    NSURL *compiledModelURL = [[dstURL URLByAppendingPathComponent:modelDirectoryName] URLByAppendingPathComponent:ETCoreMLStrings.compiledModelFileRelativePath];
    if (![fileManager moveItemAtURL:tmpCompiledModelURL toURL:compiledModelURL error:error]) {
        return nil;
    }
    
    ETCoreMLAsset *compiledModelAsset = make_asset(compiledModelURL,
                                                   @(metadata.identifier.c_str()),
                                                   fileManager,
                                                   error);
    if (!compiledModelAsset) {
        return nil;
    }
    
    ETCoreMLAssetManager *assetManager = [self createAssetManagerWithURL:dstURL error:error];
    if (!assetManager) {
        return nil;
    }
    
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    ETCoreMLModelAnalyzer *analyzer = [[ETCoreMLModelAnalyzer alloc] initWithCompiledModelAsset:compiledModelAsset
                                                                                     modelAsset:modelAsset
                                                                                 modelDebugInfo:modelDebugInfo
                                                                                       metadata:metadata
                                                                                  configuration:configuration
                                                                                   assetManager:assetManager
                                                                                          error:error];

    return analyzer;
}


@end
