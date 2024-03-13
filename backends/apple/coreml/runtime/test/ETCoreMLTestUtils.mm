//
// ETCoreMLTestUtils.mm
//
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"

#import <ETCoreMLAsset.h>

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

@end
