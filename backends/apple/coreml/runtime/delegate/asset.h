//
// ModelAsset.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import <numeric>
#import <optional>
#import <string>
#import <vector>

#import <serde_json.h>

namespace executorchcoreml {

/// A struct containing the file info.
struct FileInfo {
    /// The last modification time interval of the file since 1970 in milliseconds.
    int64_t last_modification_time_interval = 0;
    /// The file size in bytes.
    size_t size_in_bytes = 0;
    /// The relative file path from the containing package.
    std::string relative_path;
};

/// A struct containing the package info.
struct PackageInfo {
    /// The package name.
    std::string name;
    /// The file infos for the files inside the package.
    std::vector<FileInfo> file_infos;

    /// Returns the total size in bytes.
    inline size_t total_size_in_bytes() const noexcept {
        return std::accumulate(file_infos.begin(), file_infos.end(), size_t(0), [](size_t acc, const auto& info) {
            return acc + info.size_in_bytes;
        });
    }
};

/// A struct containing the asset info.
struct Asset {
    inline Asset(std::string identifier, std::string path, PackageInfo info) noexcept
        : identifier(std::move(identifier)), path(std::move(path)), package_info(info) { }

    inline Asset() noexcept { }

    /// Returns the total size in bytes.
    inline const size_t total_size_in_bytes() const noexcept { return package_info.total_size_in_bytes(); }

    /// The unique identifier.
    std::string identifier;
    /// The absolute file path for the asset.
    std::string path;
    /// The package info.
    PackageInfo package_info;

    inline std::string to_json_string() const noexcept { return serde::json::to_json_string(*this); }

    inline void from_json_string(std::string json_string) noexcept {
        serde::json::from_json_string(json_string, *this);
    }

    static std::optional<Asset>
    make(NSURL* srcURL, NSString* identifier, NSFileManager* fm, NSError* __autoreleasing* error);
};
} // namespace executorchcoreml
