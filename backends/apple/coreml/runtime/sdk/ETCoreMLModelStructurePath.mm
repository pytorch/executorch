//
// ETCoreModelStructurePath.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelStructurePath.h"

#import "objc_safe_cast.h"

namespace {
using namespace executorchcoreml::modelstructure;

enum ComponentType: uint8_t {
    program,
    function,
    operation,
    block
};

template<typename T> void append_component(NSDictionary<NSString *, id> *component, Path& path);

template<> void append_component<Path::Program>(NSDictionary<NSString *, id> * __unused component, Path& path) {
    path.append_component(Path::Program());
}

template<> void append_component<Path::Program::Function>(NSDictionary<NSString *, id> *component, Path& path) {
    NSString *name = SAFE_CAST(component[@(Path::Program::Function::kNameKeyName)], NSString);
    path.append_component(Path::Program::Function(name.UTF8String));
}

template<> void append_component<Path::Program::Block>(NSDictionary<NSString *, id> *component, Path& path) {
    NSNumber *index = SAFE_CAST(component[@(Path::Program::Block::kIndexKeyName)], NSNumber);
    NSInteger indexValue = (index != nil) ? index.integerValue : -1;
    path.append_component(Path::Program::Block(indexValue));
}

template<> void append_component<Path::Program::Operation>(NSDictionary<NSString *, id> *component, Path& path) {
    NSString *output_name = SAFE_CAST(component[@(Path::Program::Operation::kOutputKeyName)], NSString) ?: @"";
    NSCAssert(output_name.length > 0, @"Component=%@ is missing %s key.", component, Path::Program::Operation::kOutputKeyName);
    path.append_component(Path::Program::Operation(output_name.UTF8String));
}

NSDictionary<NSString *, NSNumber *> *component_types() {
    static NSDictionary<NSString *, NSNumber *> *result = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        result = @{
            @(Path::Program::kTypeName): @(ComponentType::program),
            @(Path::Program::Function::kTypeName): @(ComponentType::function),
            @(Path::Program::Block::kTypeName): @(ComponentType::block),
            @(Path::Program::Operation::kTypeName): @(ComponentType::operation)
        };
    });
    
    return result;
}

Path to_path(NSArray<NSDictionary<NSString *, id> *> *components) {
    Path path;
    NSDictionary<NSString *, NSNumber *> *types = component_types();
    for (NSDictionary<NSString *, id> *component in components) {
        NSString *type = SAFE_CAST(component[@(Path::kTypeKeyName)], NSString);
        NSCAssert(type.length > 0, @"Component=%@ is missing %s key.", component, Path::kTypeKeyName);
        switch (types[type].intValue) {
            case ComponentType::program: {
                append_component<Path::Program>(component, path);
                break;
            }
            case ComponentType::function: {
                append_component<Path::Program::Function>(component, path);
                break;
            }
            case ComponentType::block: {
                append_component<Path::Program::Block>(component, path);
                break;
            }
            case ComponentType::operation: {
                append_component<Path::Program::Operation>(component, path);
                break;
            }
            default: {
                NSCAssert(type.length == 0, @"Component=%@ has invalid type=%@.", component, type);
            }
        }
    }
    
    return path;
}

NSDictionary<NSString *, id> *to_dictionary(const Path::Program& __unused program) {
    return @{@(Path::kTypeKeyName) : @(Path::Program::kTypeName)};
}

NSDictionary<NSString *, id> *to_dictionary(const Path::Program::Function& function) {
    return @{
        @(Path::kTypeKeyName) : @(Path::Program::Function::kTypeName),
        @(Path::Program::Function::kNameKeyName) : @(function.name.c_str())
    };
}

NSDictionary<NSString *, id> *to_dictionary(const Path::Program::Block& block) {
    return @{
        @(Path::kTypeKeyName) : @(Path::Program::Block::kTypeName),
        @(Path::Program::Block::kIndexKeyName) : @(block.index)
    };
}

NSDictionary<NSString *, id> *to_dictionary(const Path::Program::Operation& operation) {
    return @{
        @(Path::kTypeKeyName) : @(Path::Program::Operation::kTypeName),
        @(Path::Program::Operation::kOutputKeyName) : @(operation.output_name.c_str())
    };
}

NSArray<NSDictionary<NSString *, id> *> *to_array(const Path& path) {
    NSMutableArray<NSDictionary<NSString *, id> *> *result = [NSMutableArray arrayWithCapacity:path.size()];
    for (const auto& component : path.components()) {
        NSDictionary<NSString *, id> *value = std::visit([](auto&& arg){
            return to_dictionary(arg);
        }, component);
        [result addObject:value];
    }
    
    return result;
}
}

@implementation ETCoreMLModelStructurePath

- (instancetype)initWithUnderlyingValue:(executorchcoreml::modelstructure::Path)underlyingValue {
    self = [super init];
    if (self) {
        _underlyingValue = std::move(underlyingValue);
    }
    
    return self;
}

- (instancetype)initWithComponents:(NSArray<NSDictionary<NSString *, id> *> *)components {
    auto underlyingValue = to_path(components);
    return [self initWithUnderlyingValue:std::move(underlyingValue)];
}

- (BOOL)isEqual:(id)object {
    if (object == self) {
        return YES;
    }
    
    if (![object isKindOfClass:self.class]) {
        return NO;
    }
    
    return _underlyingValue == ((ETCoreMLModelStructurePath *)object)->_underlyingValue;
}

- (NSUInteger)hash {
    return std::hash<executorchcoreml::modelstructure::Path>()(_underlyingValue);
}

- (instancetype)copyWithZone:(NSZone *)zone {
    return [[ETCoreMLModelStructurePath allocWithZone:zone] initWithUnderlyingValue:_underlyingValue];
}

- (nullable NSString *)operationOutputName {
    using namespace executorchcoreml::modelstructure;
    auto operation = std::get_if<Path::Program::Operation>(&(_underlyingValue.components().back()));
    if (operation == nullptr) {
        return nil;
    }
    
    return @(operation->output_name.c_str());
}

- (NSArray<NSDictionary<NSString *, id> *> *)components {
    return to_array(_underlyingValue);
}

- (NSString *)description {
    return [NSString stringWithFormat:@"<MLModelStructurePath: %p> %@", (void *)self, self.components];
}


@end
