//
// ModelAsset.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import <asset.h>

#import <optional>

#import <objc_safe_cast.h>

namespace  {

NSNumber * _Nullable is_regular_file(NSURL *url, NSError * __autoreleasing *error) {
    NSNumber *result = nil;
    if (![url getResourceValue:&result forKey:NSURLIsRegularFileKey error:error]) {
        return nil;
    }
    
    return SAFE_CAST(result, NSNumber);
}

NSNumber * _Nullable get_total_allocated_file_size(NSURL *url, NSError * __autoreleasing *error) {
    NSNumber *result = nil;
    if (![url getResourceValue:&result forKey:NSURLTotalFileAllocatedSizeKey error:error]) {
        return nil;
    }
    
    return SAFE_CAST(result, NSNumber);
}

NSNumber * _Nullable get_allocated_file_size(NSURL *url, NSError * __autoreleasing *error) {
    NSNumber *result = nil;
    if (![url getResourceValue:&result forKey:NSURLFileAllocatedSizeKey error:error]) {
        return nil;
    }
    
    return SAFE_CAST(result, NSNumber);
}

NSDate * _Nullable get_modification_date(NSURL *url, NSError * __autoreleasing *error) {
    NSDate *result = nil;
    if (![url getResourceValue:&result forKey:NSURLContentModificationDateKey error:error]) {
        return nil;
    }
    
    return SAFE_CAST(result, NSDate);
}

void set_error_from_local_error( NSError * __autoreleasing *error, NSError *local_error) {
    if (local_error && error) {
        *error = local_error;
    }
}

std::optional<executorchcoreml::PackageInfo>
get_package_info(NSURL *directory_url, NSFileManager *fm, NSError * __autoreleasing *error) {
    NSArray<NSString *> *properties = @[NSURLIsRegularFileKey, NSURLFileAllocatedSizeKey, NSURLTotalFileAllocatedSizeKey];
    
    __block NSError *local_error = nil;
    BOOL (^errorHandler)(NSURL *url, NSError *error) = ^BOOL(NSURL *url, NSError *enumeration_error) {
        local_error = enumeration_error;
        return NO;
    };
    
    NSDirectoryEnumerator *enumerator = [fm enumeratorAtURL:directory_url
                                 includingPropertiesForKeys:properties
                                                    options:NSDirectoryEnumerationProducesRelativePathURLs
                                               errorHandler:errorHandler];
    
    auto result = executorchcoreml::PackageInfo {.name = directory_url.lastPathComponent.UTF8String};
    for (NSURL *item_url in enumerator) {
        if (local_error != nil) {
            set_error_from_local_error(error, local_error);
            return std::nullopt;
        }
        
        NSURL *file_url = [NSURL fileURLWithPath:item_url.path relativeToURL:directory_url];
        if (!file_url) {
            continue;
        }
        
        if (!is_regular_file(file_url, &local_error).boolValue) {
            continue;
        }
        
        NSNumber *size = get_total_allocated_file_size(file_url, &local_error) ?: get_allocated_file_size(file_url, &local_error);
        if (!size) {
            break;
        }
        
        NSDate *last_modification_date = get_modification_date(file_url, &local_error);
        if (!last_modification_date) {
            break;
        }
        
        int64_t last_modification_time_interval = static_cast<int64_t>(last_modification_date.timeIntervalSince1970 * 1000);
        auto fileInfo = executorchcoreml::FileInfo {
            .relative_path = std::string(item_url.relativePath.UTF8String),
            .size_in_bytes = size.unsignedLongLongValue,
            .last_modification_time_interval = last_modification_time_interval
        };
        
        result.file_infos.emplace_back(std::move(fileInfo));
    }
    
    if (local_error) {
        set_error_from_local_error(error, local_error);
        return std::nullopt;
    }
    
    return result;
}

}

namespace executorchcoreml {

std::optional<Asset> Asset::make(NSURL *url,
                                 NSString *identifier,
                                 NSFileManager *fm,
                                 NSError * __autoreleasing *error) {
    auto package_info = get_package_info(url, fm, error);
    if (!package_info) {
        return std::nullopt;
    }
    
    return Asset(identifier.UTF8String, url.path.UTF8String, std::move(package_info.value()));
}
}
