//
// program_path.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import "program_path.h"

namespace {
using namespace executorchcoreml::modelstructure;

template <typename LAST, typename CURRENT>
void append_component(CURRENT component, std::vector<Path::Component>& components) {
    LAST *lastComponent = nullptr;
    if (components.size() > 0) {
        lastComponent = std::get_if<LAST>(&(components.back()));
    }
    
    NSCAssert(components.size() > 0 && lastComponent != nullptr, @"Failed to append %s component, last component is not a %s", CURRENT::kTypeName, LAST::kTypeName);
    components.emplace_back(std::move(component));
}

void append_component(Path::Program component, std::vector<Path::Component>& components) {
    NSCAssert(components.size() == 0, @"Failed to append %s component, components is not empty.", Path::Program::kTypeName);
    components.emplace_back(std::move(component));
}

void append_component(Path::Program::Function component, std::vector<Path::Component>& components) {
    append_component<Path::Program, Path::Program::Function>(std::move(component), components);
}

void append_component(Path::Program::Block component, std::vector<Path::Component>& components) {
    if (component.index >= 0) {
        append_component<Path::Program::Operation, Path::Program::Block>(std::move(component), components);
    } else {
        append_component<Path::Program::Function, Path::Program::Block>(std::move(component), components);
    }
}

void append_component(Path::Program::Operation component, std::vector<Path::Component>& components) {
    append_component<Path::Program::Block, Path::Program::Operation>(std::move(component), components);
}
}

namespace executorchcoreml {
namespace modelstructure {

const char *Path::kTypeKeyName = "Type";

const char *Path::Program::kTypeName = "Program";

const char *Path::Program::Function::kTypeName = "Function";
const char *Path::Program::Function::kNameKeyName = "Name";

const char *Path::Program::Block::kTypeName = "Block";
const char *Path::Program::Block::kIndexKeyName = "Index";

const char *Path::Program::Operation::kTypeName = "Operation";
const char *Path::Program::Operation::kOutputKeyName = "Output";

void Path::append_component(Path::Component component) noexcept {
    std::visit([&](auto&& arg){
        return ::append_component(arg, components_);
    }, component);
}
} // namespace modelstructure
} // namespace executorchcoreml
