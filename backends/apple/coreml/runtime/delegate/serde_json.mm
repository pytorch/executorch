//
// serde_json.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <serde_json.h>

#import <asset.h>
#import <objc_json_serde.h>
#import <model_metadata.h>

namespace  {
struct FileInfoKeys {
    constexpr static std::string_view kRelativePath = "relativePath";
    constexpr static std::string_view kSizeKey = "sizeInBytes";
    constexpr static std::string_view kLastModificationTimeIntervalKey = "lastModificationTimeInterval";
};

struct PackageInfoKeys {
    constexpr static std::string_view kNameKey = "name";
    constexpr static std::string_view kFileInfosKey = "fileInfos";
};

struct ModelAssetKeys {
    constexpr static std::string_view kIdentifierKey = "identifier";
    constexpr static std::string_view kPathKey = "path";
    constexpr static std::string_view kPackageInfoKey = "packageInfo";
};

struct ModelMetadataKeys {
    constexpr static std::string_view kIdentifierKey = "identifier";
    constexpr static std::string_view kInputNamesKey = "inputNames";
    constexpr static std::string_view kOutputNamesKey = "outputNames";
};
}

namespace executorchcoreml {
namespace serde {
namespace json {

template <>
struct Converter<executorchcoreml::FileInfo> {
    static id to_json(const executorchcoreml::FileInfo& file_info) {
        return @{
            to_string(FileInfoKeys::kRelativePath) : to_json_value(file_info.relative_path),
            to_string(FileInfoKeys::kSizeKey) : to_json_value(file_info.size_in_bytes),
            to_string(FileInfoKeys::kLastModificationTimeIntervalKey) :to_json_value(file_info.last_modification_time_interval)
        };
    }
    
    static void from_json(id json, executorchcoreml::FileInfo& file_info) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(FileInfoKeys::kRelativePath)], file_info.relative_path);
        from_json_value(json_dict[to_string(FileInfoKeys::kSizeKey)], file_info.size_in_bytes);
        from_json_value(json_dict[to_string(FileInfoKeys::kLastModificationTimeIntervalKey)], file_info.last_modification_time_interval);
    }
};

template <>
struct Converter<executorchcoreml::PackageInfo> {
    static id to_json(const executorchcoreml::PackageInfo& package_info) {
        return @{
            to_string(PackageInfoKeys::kNameKey) : to_json_value(package_info.name),
            to_string(PackageInfoKeys::kFileInfosKey) : to_json_value(package_info.file_infos)
        };
    }
    
    static void from_json(id json, executorchcoreml::PackageInfo& package_info) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(PackageInfoKeys::kNameKey)], package_info.name);
        from_json_value(json_dict[to_string(PackageInfoKeys::kFileInfosKey)], package_info.file_infos);
    }
};

template <>
struct Converter<executorchcoreml::Asset> {
    static id to_json(const executorchcoreml::Asset& asset) {
        return @{
            to_string(ModelAssetKeys::kIdentifierKey) : to_json_value(asset.identifier),
            to_string(ModelAssetKeys::kPathKey) : to_json_value(asset.path),
            to_string(ModelAssetKeys::kPackageInfoKey) : to_json_value(asset.package_info)
        };
    }
    
    static void from_json(id json, executorchcoreml::Asset& asset) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(ModelAssetKeys::kIdentifierKey)], asset.identifier);
        from_json_value(json_dict[to_string(ModelAssetKeys::kPathKey)], asset.path);
        from_json_value(json_dict[to_string(ModelAssetKeys::kPackageInfoKey)], asset.package_info);
    }
};

template <>
struct Converter<executorchcoreml::ModelMetadata> {
    static id to_json(const executorchcoreml::ModelMetadata& metadata) {
        return @{
            to_string(ModelMetadataKeys::kIdentifierKey) : to_json_value(metadata.identifier),
            to_string(ModelMetadataKeys::kInputNamesKey) : to_json_value(metadata.input_names),
            to_string(ModelMetadataKeys::kOutputNamesKey) :to_json_value(metadata.output_names)
        };
    }
    
    static void from_json(id json, executorchcoreml::ModelMetadata& metadata) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(ModelMetadataKeys::kIdentifierKey)], metadata.identifier);
        from_json_value(json_dict[to_string(ModelMetadataKeys::kInputNamesKey)], metadata.input_names);
        from_json_value(json_dict[to_string(ModelMetadataKeys::kOutputNamesKey)], metadata.output_names);
    }
};

std::string to_json_string(const Asset& asset) {
    id json = Converter<Asset>::to_json(asset);
    return to_json_string(json);
}

void from_json_string(const std::string& json_string, Asset& asset) {
    id json = to_json_object(json_string);
    Converter<Asset>::from_json(json, asset);
}

std::string to_json_string(const ModelMetadata& metdata) {
    id json = Converter<ModelMetadata>::to_json(metdata);
    return to_json_string(json);
}

void from_json_string(const std::string& json_string, ModelMetadata& metadata) {
    id json = to_json_object(json_string);
    Converter<ModelMetadata>::from_json(json, metadata);
}

} // namespace json
} // namespace serde
} // namespace executorchcoreml
