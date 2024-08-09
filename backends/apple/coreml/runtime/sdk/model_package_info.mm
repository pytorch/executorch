//
// model_package_info.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "model_package_info.h"

#import "ETCoreMLLogging.h"
#import "objc_json_serde.h"
#import "serde_json.h"

namespace  {
struct ModelPackageInfoKeys {
    struct Item {
        constexpr static std::string_view kAuthorKey = "author";
        constexpr static std::string_view kDescriptionKey = "description";
        constexpr static std::string_view kNameKey = "name";
        constexpr static std::string_view kPathKey = "path";
    };
    
    constexpr static std::string_view kItemInfoEntriesKey = "itemInfoEntries";
    constexpr static std::string_view kRootModelIdentifierKey = "rootModelIdentifier";
};
}

namespace executorchcoreml {
namespace serde {
namespace json {
template <>
struct Converter<ModelPackageInfo::Item> {
    static void from_json(id json, ModelPackageInfo::Item& item) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::Item::kAuthorKey)], item.author);
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::Item::kDescriptionKey)], item.description);
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::Item::kNameKey)], item.name);
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::Item::kPathKey)], item.path);
    }
};

template <>
struct Converter<ModelPackageInfo> {
    static void from_json(id json, ModelPackageInfo& package_info) {
        NSDictionary<NSString *, id> *json_dict = SAFE_CAST(json, NSDictionary);
        if (!json_dict) {
            return;
        }
        
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::kRootModelIdentifierKey)], package_info.root_model_identifier);
        from_json_value(json_dict[to_string(ModelPackageInfoKeys::kItemInfoEntriesKey)], package_info.items);
    }
};
}
}
}

namespace executorchcoreml {
std::optional<ModelPackageInfo> ModelPackageInfo::make(NSURL* model_package_url,
                                                       NSFileManager* fm,
                                                       NSError * __autoreleasing *error) noexcept {
    NSURL *manifest_url = [model_package_url URLByAppendingPathComponent:@"manifest.json"].URLByStandardizingPath;
    BOOL is_directory = NO;
    if (![fm fileExistsAtPath:manifest_url.path isDirectory:&is_directory] || is_directory) {
        ETCoreMLLogErrorAndSetNSError(error, 0, "%@ is broken, manifest doesn't exist.", model_package_url.lastPathComponent);
        return std::nullopt;
    }
    
    NSData *data = [NSData dataWithContentsOfURL:manifest_url options:NSDataReadingMappedIfSafe error:error];
    if (!data) {
        return std::nullopt;
    }
    
    id json_dictionary = [NSJSONSerialization JSONObjectWithData:data options:(NSJSONReadingOptions)0 error:error];
    if (!json_dictionary) {
        return std::nullopt;
    }
    
    ModelPackageInfo info;
    serde::json::from_json_value(json_dictionary, info);
    return info;
}
}
