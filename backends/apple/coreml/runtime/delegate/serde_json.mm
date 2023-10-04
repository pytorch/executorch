//
// serde_json.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <serde_json.h>

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
} //namespace

namespace executorchcoreml {

using json = nlohmann::json;

void to_json(json& j, const FileInfo& info) {
    j = json{
        {FileInfoKeys::kRelativePath, info.relative_path},
        {FileInfoKeys::kSizeKey, info.size_in_bytes},
        {FileInfoKeys::kLastModificationTimeIntervalKey, info.last_modification_time_interval},
    };
}

void from_json(const json& j, FileInfo& info) {
    if (j.contains(FileInfoKeys::kRelativePath)) {
        j.at(FileInfoKeys::kRelativePath).get_to(info.relative_path);
    }
    if (j.contains(FileInfoKeys::kSizeKey)) {
        j.at(FileInfoKeys::kSizeKey).get_to(info.size_in_bytes);
    }
    if (j.contains(FileInfoKeys::kLastModificationTimeIntervalKey)) {
        j.at(FileInfoKeys::kLastModificationTimeIntervalKey).get_to(info.last_modification_time_interval);
    }
}

void to_json(json& j, const PackageInfo& info) {
    j = json{
        {PackageInfoKeys::kNameKey, info.name},
        {PackageInfoKeys::kFileInfosKey, info.file_infos}
    };
}

void from_json(const json& j, PackageInfo& info) {
    if (j.contains(PackageInfoKeys::kNameKey)) {
        j.at(PackageInfoKeys::kNameKey).get_to(info.name);
    }
    if (j.contains(PackageInfoKeys::kFileInfosKey)) {
        j.at(PackageInfoKeys::kFileInfosKey).get_to(info.file_infos);
    }
}

void to_json(json& j, const Asset& asset) {
    j = json{
        {ModelAssetKeys::kIdentifierKey, asset.identifier},
        {ModelAssetKeys::kPathKey, asset.path},
        {ModelAssetKeys::kPackageInfoKey, asset.package_info}
    };
}

void from_json(const json& j, Asset& asset) {
    if (j.contains(ModelAssetKeys::kIdentifierKey)) {
        j.at(ModelAssetKeys::kIdentifierKey).get_to(asset.identifier);
    }
    if (j.contains(ModelAssetKeys::kPathKey)) {
        j.at(ModelAssetKeys::kPathKey).get_to(asset.path);
    }
    if (j.contains(ModelAssetKeys::kPackageInfoKey)) {
        j.at(ModelAssetKeys::kPackageInfoKey).get_to(asset.package_info);
    }
}

void to_json(json& j, const ModelMetadata& metadata) {
    j = json{
        {ModelMetadataKeys::kIdentifierKey, metadata.identifier},
        {ModelMetadataKeys::kInputNamesKey, metadata.input_names},
        {ModelMetadataKeys::kOutputNamesKey, metadata.output_names}
    };
}

void from_json(const json& j, ModelMetadata& metadata) {
    if (j.contains(ModelMetadataKeys::kIdentifierKey)) {
        j.at(ModelMetadataKeys::kIdentifierKey).get_to(metadata.identifier);
    }
    if (j.contains(ModelMetadataKeys::kInputNamesKey)) {
        j.at(ModelMetadataKeys::kInputNamesKey).get_to(metadata.input_names);
    }
    if (j.contains(ModelMetadataKeys::kOutputNamesKey)) {
        j.at(ModelMetadataKeys::kOutputNamesKey).get_to(metadata.output_names);
    }
}

} // namespace executorchcoreml
