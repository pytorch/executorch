//
// ETCoreMLModelDebugger.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelDebugger.h"

#import <CoreML/CoreML.h>
#import "ETCoreMLAsset.h"
#import "ETCoreMLAssetManager.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModelCompiler.h"
#import "ETCoreMLModelDebugInfo.h"
#import "ETCoreMLModelStructurePath.h"
#import "ETCoreMLPair.h"
#import "ETCoreMLStrings.h"
#import <format/MIL.pb.h>
#import <format/Model.pb.h>
#import <fstream>
#import <iostream>
#import "model_package_info.h"
#import "objc_json_serde.h"
#import <string>
#import <unordered_map>

typedef ETCoreMLPair<MLModel *, NSArray<ETCoreMLModelStructurePath *> *> DebuggableModel;

namespace {
using namespace executorchcoreml;
using namespace executorchcoreml::modelstructure;
using namespace CoreML::Specification;

NSURL * _Nullable get_model_spec_url(NSURL *model_package_url,
                                     NSFileManager *file_manager,
                                     NSError* __autoreleasing *error) {
    auto info = ModelPackageInfo::make(model_package_url, file_manager, error);
    if (!info) {
        return nil;
    }
    
    const auto& info_value = info.value();
    auto it = info_value.items.find(info_value.root_model_identifier);
    if (it == info_value.items.end()) {
        ETCoreMLLogErrorAndSetNSError(error, 0, "%@ is broken, root model info doesn't exist.", model_package_url.lastPathComponent);
        return nil;
    }
    
    auto path = it->second.path;
    if (path.empty()) {
        ETCoreMLLogErrorAndSetNSError(error, 0, "%@ is broken, root model path doesn't exist.", model_package_url.lastPathComponent);
        return nil;
    }
    
    return [model_package_url URLByAppendingPathComponent:[NSString stringWithFormat:@"Data/%s", path.c_str()]];
}

std::optional<int> index_of_output(const MILSpec::Operation& operation, const std::string& output_name) {
    for (int i = 0; i < operation.outputs_size(); i++) {
        if (operation.outputs(i).name() == output_name) {
            return i;
        }
    }
    
    return std::nullopt;
}

BOOL is_const_operation(const MILSpec::Operation& operation) {
    return operation.type() == "const";
}

BOOL is_datatype_supported_as_model_output(MILSpec::DataType datatype) {
    switch (datatype) {
        case MILSpec::DataType::INT32:
            return true;
        case MILSpec::DataType::FLOAT16:
            return true;
        case MILSpec::DataType::FLOAT32:
            return true;
        case MILSpec::DataType::FLOAT64:
            return true;
        default:
            return false;
    }
}

BOOL is_output_type_supported_as_model_output(const MILSpec::ValueType& type) {
    return type.has_tensortype() && is_datatype_supported_as_model_output(type.tensortype().datatype());
}

BOOL is_operation_output_supported_as_model_output(const MILSpec::Operation& operation) {
    if (is_const_operation(operation)) {
        return NO;
    }

    return YES;
}

const MILSpec::NamedValueType *add_output(MILSpec::Block& block, const Path& path, size_t block_component_index) {
    const auto& components = path.components();
    auto block_component = std::get_if<Path::Program::Block>(&components[block_component_index]);
    NSCAssert(block_component != nullptr, @"%@: Invalid path, component doesn't refer to a block.", NSStringFromClass(ETCoreMLModelDebugger.class));
    // Next component must be an operation.
    size_t operation_component_index = block_component_index + 1;
    auto operation_component = std::get_if<Path::Program::Operation>(&components[operation_component_index]);
    const auto& output_name = operation_component->output_name;
    
    for (int i = 0; i < block.operations_size(); i++) {
        auto& operation = *(block.mutable_operations(i));
        auto output_index = index_of_output(operation, output_name);
        if (!output_index) {
            continue;
        }
        
        if (components.size() == operation_component_index + 1) {
            const auto& output = operation.outputs(output_index.value());
            if (!is_output_type_supported_as_model_output(output.type())) {
                return nullptr;
            }
            
            block.add_outputs(output.name());
            return &output;
        }
        
        // Handle nested block.
        size_t nested_block_index = operation_component_index + 1;
        auto nested_block_component = std::get_if<Path::Program::Block>(&components[nested_block_index]);
        NSCAssert(nested_block_component != nullptr, @"%@: Invalid path, component doesn't refer to a nested block.", NSStringFromClass(ETCoreMLModelDebugger.class));
        auto& nested_block = *(operation.mutable_blocks(static_cast<int>(nested_block_component->index)));
        return add_output(nested_block, path, nested_block_index);
    }
    
    return nullptr;
}

const MILSpec::NamedValueType *add_output(MILSpec::Function& function, const Path& path, size_t function_component_index) {
    size_t block_component_index = function_component_index + 1;
    const auto& block_name = function.opset();
    auto& block = (*function.mutable_block_specializations())[block_name];
    
    return add_output(block, path, block_component_index);
}

const MILSpec::NamedValueType *add_output(MILSpec::Program& program, const Path& path) {
    size_t function_component_index = 1;
    const auto& components = path.components();
    auto function_component = std::get_if<Path::Program::Function>(&components[function_component_index]);
    NSCAssert(function_component != nullptr, @"%@: Invalid path, path doesn't refer to a function.", NSStringFromClass(ETCoreMLModelDebugger.class));
    auto& functions = *(program.mutable_functions());
    auto& function = functions[function_component->name];
    
    return add_output(function, path, function_component_index);
}

std::optional<ArrayFeatureType_ArrayDataType> to_array_datatype(MILSpec::DataType datatype,
                                                                int model_specification_version) {
    switch (datatype) {
        case MILSpec::DataType::INT32:
            return ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_INT32;
        case MILSpec::DataType::FLOAT16:
            return ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT16;
        case MILSpec::DataType::FLOAT32:
            return ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32;
        case MILSpec::DataType::FLOAT64:
            return ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_DOUBLE;
        default:
            return std::nullopt;
    }
}

const MILSpec::NamedValueType *add_output(Model& model, const Path& path) {
    NSCAssert(model.has_mlprogram(), @"%@: Model is not a ML Program.", NSStringFromClass(ETCoreMLModelDebugger.class));
    auto& program = *(model.mutable_mlprogram());
    auto output = add_output(program, path);
    if (!output) {
        return nullptr;
    }
    
    auto& description = *(model.mutable_description());
    auto& output_feature = *(description.add_output());
    output_feature.mutable_name()->assign(output->name());
    auto& multi_array_type = *(output_feature.mutable_type()->mutable_multiarraytype());
    NSCAssert(output->type().has_tensortype(), @"%@: Only a tensor type can be model output.", NSStringFromClass(ETCoreMLModelDebugger.class));
    auto tensor_type = output->type().tensortype();
    auto feature_data_type = to_array_datatype(tensor_type.datatype(), model.specificationversion());
    NSCAssert(feature_data_type.has_value(), @"%@: Unsupported datatype.", NSStringFromClass(ETCoreMLModelDebugger.class));
    multi_array_type.set_datatype(feature_data_type.value());
    
    return output;
}

void visit_program_operation(const MILSpec::Block& block,
                             const Path& block_path,
                             BOOL (^handler)(const MILSpec::Operation& operation, ETCoreMLModelStructurePath *path)) {
    for (int i = 0; i < block.operations_size(); ++i) {
        const MILSpec::Operation& operation = block.operations(i);
        Path operation_path = block_path;
        if (operation.outputs_size() == 0) {
            continue;
        }
        operation_path.append_component(Path::Program::Operation(operation.outputs(0).name()));
        if (!handler(operation, [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:operation_path])) {
            return;
        }
        
        for (int j = 0; j < operation.blocks_size(); ++j) {
            Path nested_block_path = operation_path;
            nested_block_path.append_component(Path::Program::Block(j));
            visit_program_operation(operation.blocks(j), nested_block_path, handler);
        }
    }
}

void visit_program_operation(Model& model, BOOL (^handler)(const MILSpec::Operation& operation, ETCoreMLModelStructurePath *path)) {
    const auto& functions = model.mlprogram().functions();
    for (const auto& [function_name, function] : functions) {
        Path path;
        path.append_component(Path::Program());
        path.append_component(Path::Program::Function(function_name));
        path.append_component(Path::Program::Block(-1));
        const auto& blocks = function.block_specializations();
        const auto& specialization = blocks.at(function.opset());
        visit_program_operation(specialization, path, handler);
    }
}

NSString *to_string(MLComputeUnits compute_units) {
    switch (compute_units) {
        case MLComputeUnitsAll: {
            return ETCoreMLStrings.allComputeUnitsName;
        }
        case MLComputeUnitsCPUOnly: {
            return ETCoreMLStrings.cpuComputeUnitName;
        }
        case MLComputeUnitsCPUAndGPU: {
            return ETCoreMLStrings.cpuAndGpuComputeUnitsName;
        }
        case MLComputeUnitsCPUAndNeuralEngine: {
            return ETCoreMLStrings.cpuAndNeuralEngineComputeUnitsName;
        }
        default: {
            return ETCoreMLStrings.allComputeUnitsName;
        }
    }
}

NSString *get_asset_identifier(NSString *asset_identifier,
                               MLComputeUnits compute_units,
                               NSArray<ETCoreMLModelStructurePath *> *paths) {
    size_t paths_hash = 0;
    for (ETCoreMLModelStructurePath *path in paths) {
        executorchcoreml::hash_combine(paths_hash, path.underlyingValue);
    }
    
    return [NSString stringWithFormat:@"%@_%zu_%@", asset_identifier, paths_hash, to_string(compute_units)];
}

std::unique_ptr<Model> parse_model_spec(NSURL *model_spec_url,
                                        NSError * __autoreleasing *error) {
    NSData *data = [NSData dataWithContentsOfURL:model_spec_url options:NSDataReadingMappedIfSafe error:error];
    if (!data) {
        return nullptr;
    }
    
    auto model = std::make_unique<Model>();
    if (!model->ParseFromArray(data.bytes, (int)data.length)) {
        return nullptr;
    }
    
    return model;
}

std::unique_ptr<Model> copy_model_spec(const Model& model_spec) {
    auto model_spec_copy = std::make_unique<Model>();
    model_spec_copy->CopyFrom(model_spec);
    
    return model_spec_copy;
}

void update_model_spec_version_to_include_fp16_output(Model& model_spec) {
    constexpr int minimum_spec_version_with_fp16_support = 7;
    int spec_version = MAX(model_spec.specificationversion(), minimum_spec_version_with_fp16_support);
    model_spec.set_specificationversion(spec_version);
}

NSURL * _Nullable get_compiled_model_url_with_intermediate_outputs(NSURL *model_url,
                                                                   NSURL *model_spec_url,
                                                                   const Model& model_spec,
                                                                   NSOrderedSet<NSString *> *outputNames,
                                                                   NSArray<ETCoreMLModelStructurePath *> *paths,
                                                                   NSError * __autoreleasing *error) {
    // Update model asset spec.
    auto model_spec_copy = copy_model_spec(model_spec);
    for (ETCoreMLModelStructurePath *path in paths) {
        if ([outputNames containsObject:path.operationOutputName]) {
            continue;
        }
        add_output(*model_spec_copy.get(), path.underlyingValue);
    }
    
    update_model_spec_version_to_include_fp16_output(*model_spec_copy);
    int size = model_spec_copy->ByteSize();
    NSMutableData *data = [NSMutableData dataWithLength:size];
    if (!model_spec_copy->SerializeToArray(data.mutableBytes, size)) {
        return nil;
    }
    
    if (![data writeToURL:model_spec_url options:NSDataWritingAtomic error:error]) {
        return nil;
    }
    
    return [ETCoreMLModelCompiler compileModelAtURL:model_url
                               maxWaitTimeInSeconds:(5 * 60)
                                              error:error];
}

ETCoreMLAsset * _Nullable make_asset(NSURL *asset_url,
                                     NSString *identifier,
                                     NSFileManager *fm,
                                     NSError * __autoreleasing *error) {
    auto underlying_asset = Asset::make(asset_url, identifier, fm, error);
    if (!underlying_asset) {
        return nil;
    }
    
    ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:std::move(underlying_asset.value())];
    if (![asset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    return asset;
}

NSArray<NSString *> *get_output_names(NSArray<ETCoreMLModelStructurePath *> *paths) {
    NSMutableArray<NSString *> *result = [NSMutableArray arrayWithCapacity:paths.count];
    for (ETCoreMLModelStructurePath *path in paths) {
        NSString *output_name = path.operationOutputName;
        if (output_name) {
            [result addObject:output_name];
        }
    }
    
    return result;
}

void set_model_outputs(id<MLFeatureProvider> output_features,
                       NSOrderedSet<NSString *> *output_names,
                       NSArray<MLMultiArray *> *_Nullable __autoreleasing *_Nonnull model_outputs) {
    NSMutableArray<MLMultiArray *> *values = [NSMutableArray arrayWithCapacity:output_names.count];
    for (NSString *output_name in output_names) {
        MLFeatureValue *feature_value = [output_features featureValueForName:output_name];
        NSCAssert(feature_value.multiArrayValue != nil, @"%@: Expected a multiarray value for output name=%@.",
                  NSStringFromClass(ETCoreMLModelDebugger.class),
                  output_name);
        [values addObject:feature_value.multiArrayValue];
    }
    
    *model_outputs = values;
}

void set_intermediate_outputs(id<MLFeatureProvider> output_features,
                              NSArray<ETCoreMLModelStructurePath *> *paths,
                              NSMutableDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *result) {
    for (ETCoreMLModelStructurePath *path in paths) {
        NSString *output_name = path.operationOutputName;
        if (!output_name) {
            continue;
        }
        
        MLFeatureValue *feature_value = [output_features featureValueForName:output_name];
        if (!feature_value) {
            continue;
        }
        MLMultiArray *multi_array = feature_value.multiArrayValue;
        result[path] = multi_array;
    }
}

NSArray<ETCoreMLModelStructurePath *> *get_operation_dependencies(const MILSpec::Operation &operation,
                                                                  ETCoreMLModelStructurePath *path,
                                                                  NSSet<ETCoreMLModelStructurePath *> *paths) {
    const auto& inputs = operation.inputs();
    const auto cppPath = path.underlyingValue;
    NSMutableArray<ETCoreMLModelStructurePath *> *deps = [NSMutableArray arrayWithCapacity:inputs.size()];
    for (const auto& [_, arg] : inputs) {
        const auto& bindings = arg.arguments();
        for (const auto& binding : bindings) {
            if (binding.has_value()) {
                continue;
            }

            const auto& name = binding.name();
            auto dep = cppPath;
            dep.remove_last_component();
            dep.append_component(Path::Program::Operation(name));
            ETCoreMLModelStructurePath *path = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:dep];
            if ([paths containsObject:path]) {
                [deps addObject:path];
            }
        }
    }

    return deps;
}

NSDictionary<NSString *, NSArray<ETCoreMLModelStructurePath *> *> *get_debug_handle_to_operation_paths_map(ETCoreMLModelDebugInfo *debug_info) {
    NSUInteger capacity = debug_info.pathToDebugHandlesMap.count;
    NSMutableDictionary<NSString *, NSMutableArray<ETCoreMLModelStructurePath *> *> *result = [NSMutableDictionary dictionaryWithCapacity:capacity];
    [debug_info.pathToDebugHandlesMap enumerateKeysAndObjectsUsingBlock:^(ETCoreMLModelStructurePath *path,
                                                                          NSArray<NSString *> *debug_handles,
                                                                          BOOL * _Nonnull __unused stop) {
        for (NSString *debug_handle in debug_handles) {
            NSMutableArray<ETCoreMLModelStructurePath *> *paths = result[debug_handle];
            if (!paths) {
                paths = [NSMutableArray array];
                result[debug_handle] = paths;
            }

            [paths addObject:path];
        }

    }];

    return result;
}

BOOL is_node_terminal_node(ETCoreMLModelStructurePath *node,
                           NSArray<ETCoreMLModelStructurePath *> *nodes,
                           NSDictionary<ETCoreMLModelStructurePath *, NSArray<ETCoreMLModelStructurePath *> *> *dependencies) {
    NSMutableSet<ETCoreMLModelStructurePath *> *nodes_dependencies = [NSMutableSet set];
    for (ETCoreMLModelStructurePath *current_node in nodes) {
        if ([current_node isEqual:node]) {
            continue;
        }
        NSArray<ETCoreMLModelStructurePath *> *node_dependencies = dependencies[current_node];
        if (node_dependencies.count > 0) {
            [nodes_dependencies addObjectsFromArray:node_dependencies];
        }
    }

    return ![nodes_dependencies containsObject:node];
}

ETCoreMLModelStructurePath *_Nullable find_terminal_node_from_nodes(NSArray<ETCoreMLModelStructurePath *> *nodes,
                                                                    NSDictionary<ETCoreMLModelStructurePath *, NSArray<ETCoreMLModelStructurePath *> *> *dependencies) {
    if (nodes.count < 2) {
        return nodes.firstObject;
    }

    for (ETCoreMLModelStructurePath *node in nodes) {
        if (is_node_terminal_node(node, nodes, dependencies)) {
            return node;
        }
    }

    return nil;
}

NSDictionary<ETCoreMLModelStructurePath *, NSString *> *get_operation_path_to_debug_symbol_map(ETCoreMLModelDebugInfo *model_debug_info,
                                                                                               NSDictionary<NSString *, NSArray<ETCoreMLModelStructurePath *> *> *debug_handle_to_operation_paths_map,
                                                                                               NSDictionary<ETCoreMLModelStructurePath *, NSArray<ETCoreMLModelStructurePath *> *> *dependencies) {
    // When decomposing an EXIR operation into a MIL graph, it is essential to report the output of the terminal node of the MIL graph.
    // This output corresponds directly to the output of the original EXIR operation.
    NSUInteger capacity = debug_handle_to_operation_paths_map.count;
    NSMutableDictionary<ETCoreMLModelStructurePath *, NSString *> *operation_path_to_debug_symbol_map = [NSMutableDictionary dictionaryWithCapacity:capacity];
    [debug_handle_to_operation_paths_map enumerateKeysAndObjectsUsingBlock:^(NSString *debug_handle,
                                                                             NSArray<ETCoreMLModelStructurePath *> *operation_paths,
                                                                             BOOL * __unused stop) {
        ETCoreMLModelStructurePath *operation_path = find_terminal_node_from_nodes(operation_paths, dependencies);
        NSString *debug_symbol = (operation_path != nil) ? model_debug_info.pathToDebugSymbolMap[operation_path] : nil;
        if (debug_symbol) {
            operation_path_to_debug_symbol_map[operation_path] = debug_symbol;
        }

    }];

    return operation_path_to_debug_symbol_map;
}

}

@interface ETCoreMLModelDebugger ()
/// The model output names.
@property (readonly, copy, nonatomic) NSOrderedSet<NSString *> *outputNames;
/// The model asset.
@property (readonly, copy, nonatomic) ETCoreMLAsset *modelAsset;
/// The model debug info.
@property (readonly, copy, nonatomic, nullable) ETCoreMLModelDebugInfo *modelDebugInfo;
/// The asset manager.
@property (readonly, copy, nonatomic) ETCoreMLAssetManager *assetManager;
/// The model configuration.
@property (readonly, strong, nonatomic) MLModelConfiguration *configuration;
/// The url to the model specification.
@property (readonly, copy, nonatomic) NSURL *modelSpecURL;

@end

@implementation ETCoreMLModelDebugger {
    std::unique_ptr<Model> _modelSpec;
}

- (nullable instancetype)initWithModelAsset:(ETCoreMLAsset *)modelAsset
                             modelDebugInfo:(nullable ETCoreMLModelDebugInfo *)modelDebugInfo
                                outputNames:(NSOrderedSet<NSString *> *)outputNames
                              configuration:(MLModelConfiguration *)configuration
                               assetManager:(ETCoreMLAssetManager *)assetManager
                                      error:(NSError * __autoreleasing *)error {
    if (![modelAsset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    NSFileManager *fileManager = [[NSFileManager alloc] init];
    NSURL *modelSpecURL = get_model_spec_url(modelAsset.contentURL, fileManager, error);
    if (!modelSpecURL) {
        return nil;
    }
    
    auto modelSpec = parse_model_spec(modelSpecURL, error);
    if (!modelSpec) {
        return nil;
    }

    __block NSMutableDictionary<ETCoreMLModelStructurePath *, NSArray<ETCoreMLModelStructurePath *> *> *dependencies = [NSMutableDictionary dictionary];
    __block NSMutableArray<ETCoreMLModelStructurePath *> *operationPaths = [NSMutableArray array];
    __block NSMutableSet<ETCoreMLModelStructurePath *> *allOperationPaths = [NSMutableSet set];
    visit_program_operation(*modelSpec, ^BOOL(const MILSpec::Operation &operation, ETCoreMLModelStructurePath *operationPath) {
        dependencies[operationPath] = get_operation_dependencies(operation, operationPath, allOperationPaths);
        [allOperationPaths addObject:operationPath];
        if (is_operation_output_supported_as_model_output(operation)) {
            [operationPaths addObject:operationPath];
        }

        return YES;
    });


    NSDictionary<NSString *, NSArray<ETCoreMLModelStructurePath *> *> *debugHandleToOperationPathsMap = get_debug_handle_to_operation_paths_map(modelDebugInfo);

    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *operationPathToDebugSymbolMap = get_operation_path_to_debug_symbol_map(modelDebugInfo,
                                                                                                                                   debugHandleToOperationPathsMap,
                                                                                                                                   dependencies);

    self = [super init];
    if (self) {
        _modelAsset = modelAsset;
        _configuration = configuration;
        _outputNames = [outputNames copy];
        _assetManager = assetManager;
        _modelSpec = std::move(modelSpec);
        _modelSpecURL = modelSpecURL;
        _operationPaths = operationPaths;
        _operationPathToDebugSymbolMap = operationPathToDebugSymbolMap;
        _modelDebugInfo = modelDebugInfo;
    }
    
    return self;
}

- (nullable ETCoreMLAsset *)compiledModelAssetWithOutputsAtPaths:(NSArray<ETCoreMLModelStructurePath *> *)paths
                                                           error:(NSError* __autoreleasing *)error {
    NSString *identifier = get_asset_identifier(self.modelAsset.identifier,
                                                self.configuration.computeUnits,
                                                paths);
    NSError *localError = nil;
    ETCoreMLAsset *compiledAsset = [self.assetManager assetWithIdentifier:identifier error:&localError];
    if (compiledAsset) {
        return compiledAsset;
    }
    
    if (localError) {
        ETCoreMLLogError(localError,
                         "%@: Failed to retrieve asset with identifier=%@",
                         NSStringFromClass(ETCoreMLModelDebugger.class),
                         identifier);
    }
    
    NSURL *compiledModelURL = get_compiled_model_url_with_intermediate_outputs(self.modelAsset.contentURL,
                                                                               self.modelSpecURL,
                                                                               *(_modelSpec.get()),
                                                                               self.outputNames,
                                                                               paths,
                                                                               error);
    if (!compiledModelURL) {
        return nil;
    }
    
    compiledAsset = [self.assetManager storeAssetAtURL:compiledModelURL
                                        withIdentifier:identifier
                                                 error:&localError];
    
    if (compiledAsset) {
        return compiledAsset;
    }
    
    if (localError) {
        ETCoreMLLogError(localError, 
                         "%@: Failed to store asset with identifier=%@",
                         NSStringFromClass(ETCoreMLModelDebugger.class),
                         identifier);
    }
    
    return make_asset(compiledModelURL, identifier, self.assetManager.fileManager, error);
}

- (nullable NSArray<DebuggableModel *> *)_modelsWithOutputsOfOperationsAtPath:(NSArray<ETCoreMLModelStructurePath *> *)paths
                                                                        error:(NSError* __autoreleasing *)error {
    if (paths.count == 0) {
        return @[];
    }
    
    ETCoreMLAsset *compiledAsset = [self compiledModelAssetWithOutputsAtPaths:paths error:error];
    if (!compiledAsset) {
        return nil;
    }
    
    NSError *localError = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:compiledAsset.contentURL
                                       configuration:self.configuration
                                               error:&localError];
    if (model) {
        DebuggableModel *pair = [[ETCoreMLPair alloc] initWithFirst:model second:paths];
        return @[pair];
    }
    
    if (localError) {
        ETCoreMLLogError(localError, "%@: Failed to load model with outputs=%@",
                         NSStringFromClass(ETCoreMLModelDebugger.class),
                         get_output_names(paths));
    }
    
    if ([self.assetManager removeAssetWithIdentifier:compiledAsset.identifier error:&localError]) {
        ETCoreMLLogError(localError, "%@: Failed to remove compiled asset with identifier=%@",
                         NSStringFromClass(ETCoreMLModelDebugger.class),
                         compiledAsset.identifier);
    }
    
    if (paths.count == 1) {
        *error = localError;
        return nil;
    }
    
    // There is a chance that the model compilation fails because of the number of outputs. In this case, we divide the paths into two and try again.
    NSArray<ETCoreMLModelStructurePath *> *leftPaths = [paths subarrayWithRange:NSMakeRange(0, paths.count/2)];
    NSArray<ETCoreMLModelStructurePath *> *rightPaths = [paths subarrayWithRange:NSMakeRange(paths.count/2, paths.count - paths.count/2)];
    NSArray<DebuggableModel *> *leftModels = [self modelsWithOutputsOfOperationsAtPath:leftPaths error:&localError];
    NSArray<DebuggableModel *> *rightModels = [self modelsWithOutputsOfOperationsAtPath:rightPaths error:&localError];
    if (leftModels.count == 0 && rightModels.count == 0) {
        *error = localError;
        return nil;
    }
    
    NSArray<DebuggableModel *> *models = [(leftModels == nil ? @[] : leftModels) arrayByAddingObjectsFromArray:(rightModels == nil ? @[] : rightModels)];
    return models;
}

- (nullable NSArray<DebuggableModel *> *)modelsWithOutputsOfOperationsAtPath:(NSArray<ETCoreMLModelStructurePath *> *)paths
                                                                       error:(NSError* __autoreleasing *)error {
    @autoreleasepool {
        return [self _modelsWithOutputsOfOperationsAtPath:paths error:error];
    }
}

- (nullable ETCoreMLModelOutputs *)outputsOfOperationsAtPaths:(NSArray<ETCoreMLModelStructurePath *> *)paths
                                                      options:(MLPredictionOptions *)options
                                                       inputs:(id<MLFeatureProvider>)inputs
                                                 modelOutputs:(NSArray<MLMultiArray *> *_Nullable __autoreleasing *_Nonnull)modelOutputs
                                                        error:(NSError* __autoreleasing *)error {
    NSArray<MLMultiArray *> *lModelOutputs = nil;
    NSMutableDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *result = [NSMutableDictionary dictionaryWithCapacity:paths.count];
    @autoreleasepool {
        NSArray<DebuggableModel *> *models = [self modelsWithOutputsOfOperationsAtPath:paths error:error];
        if (!models) {
            return nil;
        }
        
        for (DebuggableModel *pair in models) {
            id<MLFeatureProvider> outputFeatures = [pair.first predictionFromFeatures:inputs options:options error:error];
            set_intermediate_outputs(outputFeatures, paths, result);
            if (modelOutputs) {
                set_model_outputs(outputFeatures, self.outputNames, &lModelOutputs);
            }
        }
    }
    
    if (modelOutputs) {
        *modelOutputs = lModelOutputs;
    }
    
    return result;
}

@end
