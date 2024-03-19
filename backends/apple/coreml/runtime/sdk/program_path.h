//
// program_path.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <string>
#import <variant>
#import <vector>

#import "hash_util.h"

namespace executorchcoreml {
namespace modelstructure {
class Path final {
public:
    static const char* kTypeKeyName;
    struct Program {
        /// The type name, used for serializing/deserializing Program type.
        static const char* kTypeName;
        struct Function {
            /// The type name, used for serializing/deserializing Function type.
            static const char* kTypeName;
            /// The name key name, used for serializing/deserializing function name.
            static const char* kNameKeyName;
            /// Name of the function.
            std::string name;

            Function(std::string name) : name(std::move(name)) { }

            inline bool operator==(const Function& rhs) const noexcept { return name == rhs.name; }

            inline bool operator<(const Function& rhs) const noexcept { return name < rhs.name; }
        };

        struct Block {
            /// The type name, used for serializing/deserializing Block type.
            static const char* kTypeName;
            /// The index key name, used for serializing/deserializing index.
            static const char* kIndexKeyName;
            /// Index of the block.
            int64_t index;

            Block(int64_t index) : index(index) { }

            inline bool operator==(const Block& rhs) const noexcept { return index == rhs.index; }

            inline bool operator<(const Block& rhs) const noexcept { return index < rhs.index; }
        };

        struct Operation {
            /// The type name, used for serializing/deserializing Operation type.
            static const char* kTypeName;
            /// The output key name, used for serializing/deserializing the Operation output.
            static const char* kOutputKeyName;
            /// Output name.
            std::string output_name;

            Operation(std::string output_name) : output_name(std::move(output_name)) { }

            inline bool operator==(const Operation& rhs) const noexcept { return output_name == rhs.output_name; }

            inline bool operator<(const Operation& rhs) const noexcept { return output_name < rhs.output_name; }
        };

        inline bool operator==(const Program& __unused rhs) const noexcept { return true; }
    };

    using Component = std::variant<Program, Program::Function, Program::Block, Program::Operation>;

    /// Appends a component to the path.
    void append_component(Component component) noexcept;

    /// Removes the last component.
    inline void remove_last_component() noexcept { components_.pop_back(); }

    /// Removes the first component.
    inline void remove_first_component() noexcept { components_.erase(components_.begin()); }

    /// Returns the number of components.
    inline size_t size() const noexcept { return components_.size(); }

    /// Returns components.
    inline const std::vector<Component>& components() const noexcept { return components_; }

    const Component& operator[](size_t index) const { return components_[index]; }

    inline bool operator==(const Path& rhs) const noexcept { return components() == rhs.components(); }

private:
    std::vector<Component> components_;
};
}
}

namespace std {
using namespace executorchcoreml::modelstructure;

template <> struct hash<Path::Program> {
    inline size_t operator()(const Path::Program __unused& program) const { return typeid(Path::Program).hash_code(); }
};

template <> struct hash<Path::Program::Block> {
    inline size_t operator()(const Path::Program::Block& block) const { return hash<int64_t>()(block.index); }
};

template <> struct hash<Path::Program::Function> {
    inline size_t operator()(const Path::Program::Function& function) const {
        return hash<std::string>()(function.name);
    }
};

template <> struct hash<Path::Program::Operation> {
    inline size_t operator()(const Path::Program::Operation& operation) const {
        return hash<std::string>()(operation.output_name);
    }
};

template <> struct hash<Path> {
    inline size_t operator()(const Path& path) const { return executorchcoreml::container_hash(path.components()); }
};
} // namespace std
