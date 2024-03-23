//
// ETCoreMLAsset.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLAsset.h>

#import <fcntl.h>
#import <os/lock.h>
#import <stdio.h>
#import <system_error>

#import <objc_safe_cast.h>

namespace  {
using namespace executorchcoreml;

NSDate * _Nullable get_content_modification_date(NSURL *url, NSError * __autoreleasing *error) {
    NSDate *result = nil;
    if (![url getResourceValue:&result forKey:NSURLContentModificationDateKey error:error]) {
        return nil;
    }
    
    return SAFE_CAST(result, NSDate);
}

bool is_asset_valid(const Asset& asset) {
    NSURL *asset_url = [NSURL fileURLWithPath:@(asset.path.c_str())];
    for (const auto& file_info : asset.package_info.file_infos) {
        NSError *local_error = nil;
        const std::string& relative_path = file_info.relative_path;
        NSURL *file_url = [asset_url URLByAppendingPathComponent:@(relative_path.c_str())];
        
        NSDate *last_modification_date = get_content_modification_date(file_url, &local_error);
        if (!last_modification_date) {
            return false;
        }
        
        int64_t last_modification_time_interval = static_cast<int64_t>(last_modification_date.timeIntervalSince1970 * 1000);
        if (last_modification_time_interval != file_info.last_modification_time_interval) {
            return false;
        }
    }
    
    return true;
}

void set_error_from_error_code(const std::error_code& cppError, NSError * __autoreleasing *error) {
    if (!error || !cppError) {
        return;
    }
    
    NSString *message = @(cppError.message().c_str());
    NSString *domain =  @(cppError.category().name());
    NSInteger code = cppError.value();
    NSError *localError = [NSError errorWithDomain:domain code:code userInfo:@{NSLocalizedDescriptionKey : message}];
    *error = localError;
}
} //namespace

@implementation ETCoreMLAsset {
    executorchcoreml::Asset _backingAsset;
    std::vector<std::unique_ptr<FILE, decltype(&fclose)>> _openFiles;
    os_unfair_lock _lock;
}

- (instancetype)initWithBackingAsset:(executorchcoreml::Asset)backingAsset {
    self = [super init];
    if (self) {
        _isValid = static_cast<BOOL>(is_asset_valid(backingAsset));
        _identifier = @(backingAsset.identifier.c_str());
        _contentURL = [NSURL fileURLWithPath:@(backingAsset.path.c_str())];
        _totalSizeInBytes = backingAsset.total_size_in_bytes();
        _backingAsset = std::move(backingAsset);
    }
    
    return self;
}

- (void)dealloc {
    [self close];
}

- (BOOL)_keepAliveAndReturnError:(NSError * __autoreleasing *)error {
    if (!_isValid) {
        return NO;
    }
    
    const auto& fileInfos = _backingAsset.package_info.file_infos;
    if (_openFiles.size() == fileInfos.size()) {
        return YES;
    }
    
    std::vector<std::unique_ptr<FILE, decltype(&fclose)>> openFiles;
    for (const auto& fileInfo : fileInfos) {
        NSURL *fileURL = [NSURL fileURLWithPath:@(fileInfo.relative_path.c_str()) relativeToURL:self.contentURL];
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(fileURL.path.UTF8String, "rb"), fclose);
        if (file == nullptr) {
            ::set_error_from_error_code(std::error_code(errno, std::generic_category()), error);
            break;
        }
        openFiles.emplace_back(std::move(file));
    }
    
    BOOL success = (openFiles.size() == fileInfos.size());
    if (success) {
        _openFiles = std::move(openFiles);
    }
    
    return success;
}

- (BOOL)isAlive {
    BOOL result = NO;
    {
        os_unfair_lock_lock(&_lock);
        const auto& fileInfos = _backingAsset.package_info.file_infos;
        result = (_openFiles.size() == fileInfos.size());
        os_unfair_lock_unlock(&_lock);
    }
    
    return result;
}

- (BOOL)keepAliveAndReturnError:(NSError * __autoreleasing *)error {
    BOOL result = NO;
    {
        os_unfair_lock_lock(&_lock);
        result = [self _keepAliveAndReturnError:error];
        os_unfair_lock_unlock(&_lock);
    }
    
    return result;
}

- (BOOL)prewarmAndReturnError:(NSError * __autoreleasing *)error {
    std::vector<int> fds;
    {
        os_unfair_lock_lock(&_lock);
        if ([self _keepAliveAndReturnError:error]) {
            for (const auto& file : _openFiles) {
                fds.emplace_back(fileno(file.get()));
            }
        }
        os_unfair_lock_unlock(&_lock);
    }
    
    const auto& fileInfos = _backingAsset.package_info.file_infos;
    if (fds.size() != fileInfos.size()) {
        return NO;
    }
    
    for (size_t i = 0; i < fileInfos.size(); i++) {
        const auto& fileInfo = fileInfos[i];
        size_t sizeInBytes = fileInfo.size_in_bytes;
        struct radvisory advisory = { .ra_offset = 0, .ra_count = (int)sizeInBytes };
        int fd = fds[i];
        int status = fcntl(fd, F_RDADVISE, &advisory);
        if (status == -1) {
            ::set_error_from_error_code(std::error_code(errno, std::system_category()), error);
            return NO;
        }
    }
    
    return YES;
}

- (void)close {
    os_unfair_lock_lock(&_lock);
    _openFiles.clear();
    os_unfair_lock_unlock(&_lock);
}

@end
