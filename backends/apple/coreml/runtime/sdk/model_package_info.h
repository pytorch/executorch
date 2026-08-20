//
// model_package_info.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <Foundation/Foundation.h>

#import "serde_json.h"
#import <string>
#import <unordered_map>

namespace executorchcoreml {
/// A struct containing the info of a `mlpackage`.
struct ModelPackageInfo {
    struct Item {
        /// The item author.
        std::string author;
        /// The item description.
        std::string description;
        /// The item name.
        std::string name;
        /// The item path.
        std::string path;
    };
    /// The identifier of the root model item. An entry for the name must exist in the `items`.
    std::string root_model_identifier;
    /// The items in the `mlpackage`. This is populated by parsing the `mlpackage`'s manifest file.
    std::unordered_map<std::string, Item> items;

    static std::optional<ModelPackageInfo>
    make(NSURL* url, NSFileManager* fm, NSError* __autoreleasing* error) noexcept;
};
}
