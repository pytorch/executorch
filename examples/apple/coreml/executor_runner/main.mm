//
// main.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>
#import <chrono>
#import <coreml_backend/delegate.h>
#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/runtime/executor/method.h>
#import <executorch/runtime/executor/program.h>
#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/platform/runtime.h>
#import <executorch/devtools/etdump/etdump_flatcc.h>
#import <memory>
#import <numeric>
#import <string>

static inline id check_class(id obj, Class cls) {
    return [obj isKindOfClass:cls] ? obj : nil;
}

#define SAFE_CAST(Object, Type) ((Type *)check_class(Object, [Type class]))

using executorch::etdump::ETDumpGen;
using executorch::etdump::ETDumpResult;
using executorch::extension::FileDataLoader;
using executorch::runtime::DataLoader;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
using torch::executor::CoreMLBackendDelegate;

static constexpr size_t kRuntimeMemorySize = 16 * 1024U * 1024U; // 16 MB

namespace {

struct Args {
    std::string model_path;
    std::string etdump_path = "etdump.etdp";
    std::string debug_buffer_path = "debug_buffer.bin";
    size_t debug_buffer_size = 1024 * 1024;
    size_t iterations = 1;
    bool purge_models_cache = false;
    bool dump_model_outputs = false;
    bool dump_intermediate_outputs = false;
    bool profile_model = false;

    Args(NSDictionary<NSString *, NSString *> *params) {
        {
            NSString *value = SAFE_CAST(params[@"--model_path"], NSString);
            if (value.length > 0) {
                model_path = value.UTF8String;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--etdump_path"], NSString);
            if (value.length > 0) {
                etdump_path = value.UTF8String;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--debug_buffer_path"], NSString);
            if (value.length > 0) {
                debug_buffer_path = value.UTF8String;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--iterations"], NSString);
            if (value.length > 0) {
                iterations = value.integerValue;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--debug_buffer_size"], NSString);
            if (value.length > 0) {
                debug_buffer_size = value.integerValue;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--purge_models_cache"], NSString);
            if (value.length > 0) {
                purge_models_cache = value.boolValue;
            }
        }
        {
            if (params[@"--profile_model"] != nil) {
                profile_model = true;
            }
        }
        {
            if (params[@"--dump_intermediate_outputs"] != nil) {
                dump_intermediate_outputs = true;
            }
        }
        {
            if (params[@"--dump_model_outputs"] != nil) {
                dump_model_outputs = true;
            }
        }
    }
};

NSString *clean_string(NSString *value) {
    return [value stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
}

NSSet<NSString *> *all_keys() {
    return [NSSet setWithArray:@[
        @"--model_path",
        @"--iterations",
        @"--purge_models_cache",
        @"--etdump_path",
        @"--debug_buffer_path",
        @"--debug_buffer_size",
        @"--dump_intermediate_outputs",
        @"--dump_model_outputs",
        @"--profile_model"
    ]];
}

Args parse_command_line_args(NSArray<NSString *> *args) {
    NSMutableDictionary<NSString *, NSString *> *params = [[NSMutableDictionary alloc] initWithCapacity:args.count];
    NSMutableSet<NSString *> *keys = [all_keys() mutableCopy];
    NSMutableString *values = [NSMutableString string];
    NSString *key = @"";
    for (NSString *arg in args) {
        NSString *value = clean_string(arg);
        if (![keys containsObject:value]) {
            if (value.length == 0) {
                continue;
            }
            if (values.length > 0) {
                [values appendString:@"\t"];
            }
            [values appendString:value];
            continue;
        }
        [keys removeObject:value];
        params[key] = values;
        key = value;
        values = [NSMutableString string];
    }

    if (key.length > 0) {
        params[key] = values.length > 0 ? clean_string(values.copy) : @"";
    }

    return Args(params);
}

NSData * _Nullable read_data(const std::string& file_path) {
    NSError *localError = nil;
    NSURL *url = [NSURL fileURLWithPath:@(file_path.c_str())];
    NSData *data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:&localError];
    ET_CHECK_MSG(data != nil, "Failed to read data from path=%s", file_path.c_str());
    return data;
}

class DataLoaderImpl final : public DataLoader {
public:
    DataLoaderImpl(const std::string& filePath)
    :data_(read_data(filePath))
    {}

    Result<FreeableBuffer> load(size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segment_info) const override {
        NSData *subdata = [data_ subdataWithRange:NSMakeRange(offset, size)];
        return FreeableBuffer(subdata.bytes, size, nullptr);
    }

    Result<size_t> size() const override {
        return data_.length;
    }

private:
    NSData * const data_;
};

using Buffer = std::vector<uint8_t>;

std::unique_ptr<Program> make_program(DataLoader *data_loader) {
    auto program = Program::load(data_loader);
    if (!program.ok()) {
        return nullptr;
    }

    return std::make_unique<Program>(std::move(program.get()));
}

Result<std::string> get_method_name(Program *program) {
    const auto methodName = program->get_method_name(0);
    if (!methodName.ok()) {
        return Error::InvalidProgram;
    }

    return std::string(methodName.get());
}

Result<std::vector<Buffer>>
get_planned_buffers(const std::string& method_name, Program *program) {
    auto method_meta = program->method_meta(method_name.c_str());
    if (!method_meta.ok()) {
        return Error::InvalidProgram;
    }

    std::vector<std::vector<uint8_t>> buffers;
    buffers.reserve(method_meta->num_memory_planned_buffers());
    for (size_t bufferID = 0; bufferID < method_meta->num_memory_planned_buffers(); ++bufferID) {
        auto buffer_size = method_meta->memory_planned_buffer_size(bufferID);
        std::vector<uint8_t> data(buffer_size.get(), 0);
        buffers.emplace_back(std::move(data));
    }

    return buffers;
}

std::vector<Span<uint8_t>> to_spans(std::vector<Buffer>& buffers) {
    std::vector<Span<uint8_t>> result;
    result.reserve(buffers.size());
    for (auto& buffer : buffers) {
        result.emplace_back(buffer.data(), buffer.size());
    }

    return result;
}

Result<std::vector<Buffer>> prepare_input_tensors(Method& method) {
    MethodMeta method_meta = method.method_meta();
    size_t num_inputs = method_meta.num_inputs();
    std::vector<std::vector<uint8_t>> buffers;
    for (size_t i = 0; i < num_inputs; i++) {
        Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
        if (!tensor_meta.ok()) {
            ET_LOG(Info, "Skipping non-tensor input %zu", i);
            continue;
        }
        Buffer buffer(tensor_meta->nbytes(), 0);
        auto sizes = tensor_meta->sizes();
        exec_aten::TensorImpl tensor_impl(tensor_meta->scalar_type(), std::size(sizes), const_cast<int *>(sizes.data()), buffer.data());
        exec_aten::Tensor tensor(&tensor_impl);
        EValue input_value(std::move(tensor));
        Error err = method.set_input(input_value, i);
        if (err != Error::Ok) {
            ET_LOG(Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
            return err;
        }
        buffers.emplace_back(std::move(buffer));
    }

    return buffers;
}

double calculate_mean(const std::vector<double>& durations) {
    if (durations.size() == 0) {
        return 0.0;
    }

    return std::accumulate(durations.begin(), durations.end(), 0.0)/durations.size();
}

Error execute_method(Method *method, size_t n, std::vector<double>& durations) {
    Error status = Error::Ok;
    durations.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        auto start_time = std::chrono::steady_clock::now();
        status = method->execute();
        auto current_time = std::chrono::steady_clock::now();
        if (status != Error::Ok) {
            break;
        }
        auto diff = current_time - start_time;
        durations.emplace_back(std::chrono::duration<double, std::milli>(diff).count());
    }

    return status;
}

bool is_model_analysis_enabled(const Args& args) {
    return args.profile_model || args.dump_model_outputs || args.dump_intermediate_outputs;
}

std::unique_ptr<ETDumpGen> make_etdump_gen(Buffer& debug_buffer, const Args& args) {
    if (!is_model_analysis_enabled(args)) {
        return nullptr;
    }

    auto etdump_gen = std::make_unique<ETDumpGen>();
    debug_buffer.resize(args.debug_buffer_size);
    if (args.dump_intermediate_outputs || args.dump_model_outputs) {
        debug_buffer.resize(args.debug_buffer_size);
        ET_LOG(Info, args.dump_model_outputs ? "Logging model outputs." : "Logging intermediate outputs.");
        Span<uint8_t> debug_buffer_span(debug_buffer.data(), debug_buffer.size());
        etdump_gen->set_debug_buffer(debug_buffer_span);
        etdump_gen->set_event_tracer_debug_level(args.dump_model_outputs ? EventTracerDebugLogLevel::kProgramOutputs : EventTracerDebugLogLevel::kIntermediateOutputs);
    }

    return etdump_gen;
}

void dump_etdump_gen(ETDumpGen *etdump_gen, const Buffer& debug_buffer, const Args& args) {
    ETDumpResult result = (etdump_gen != nullptr) ? etdump_gen->get_etdump_data() : ETDumpResult{.buf = nullptr, .size = 0};
    if (result.size == 0) {
        return;
    }

    FILE *ptr = fopen(args.etdump_path.c_str(), "wb");
    fwrite(result.buf, 1, result.size, ptr);
    fclose(ptr);
    ET_LOG(Info, "Profiling result saved at path = %s", args.etdump_path.c_str());
    if (args.dump_intermediate_outputs || args.dump_model_outputs) {
        ET_LOG(Info, "Debug buffer size = %zu", result.size);
        FILE *ptr = fopen(args.debug_buffer_path.c_str(), "wb");
        fwrite(debug_buffer.data(), 1, debug_buffer.size(), ptr);
        fclose(ptr);
        ET_LOG(Info, "Debug result saved at path = %s", args.etdump_path.c_str());
    }
}

}

int main(int argc, char * argv[]) {
    @autoreleasepool {
        executorch::runtime::runtime_init();

        auto args = parse_command_line_args([[NSProcessInfo processInfo] arguments]);
        if (args.purge_models_cache) {
            ET_LOG(Info, "Purging models cache");
            auto delegate = CoreMLBackendDelegate::get_registered_delegate();
            delegate->purge_models_cache();
        }

        if (args.model_path.empty()) {
            ET_LOG(Error, "Model path is empty.");
            return EXIT_FAILURE;
        }

        NSURL *model_url = [NSURL fileURLWithPath:@(args.model_path.c_str())];
        ET_CHECK_MSG(model_url != nil, "Model path=%s is invalid", args.model_path.c_str());

        auto data_loader = std::make_unique<DataLoaderImpl>(model_url.path.UTF8String);
        auto program = ::make_program(data_loader.get());
        ET_CHECK_MSG(program != nil, "Failed to load program from path=%s", args.model_path.c_str());

        auto method_name = get_method_name(program.get());
        ET_CHECK_MSG(method_name.ok(), "Failed to get method name from program=%p", program.get());

        auto planned_buffers = get_planned_buffers(method_name.get(), program.get());
        Buffer method_buffer(kRuntimeMemorySize, 0);
        MemoryAllocator method_allocator(static_cast<int32_t>(method_buffer.size()), method_buffer.data());
        auto spans = to_spans(planned_buffers.get());
        HierarchicalAllocator planned_allocator(Span<Span<uint8_t>>(reinterpret_cast<Span<uint8_t> *>(spans.data()), spans.size()));
        MemoryManager memory_manager(&method_allocator, &planned_allocator);

        Buffer debug_buffer;
        auto etdump_gen = ::make_etdump_gen(debug_buffer, args);

        auto load_start_time = std::chrono::steady_clock::now();
        auto method = program->load_method(method_name.get().c_str(), &memory_manager, (EventTracer *)etdump_gen.get());
        auto load_duration = std::chrono::steady_clock::now() - load_start_time;
        ET_LOG(Info, "Load duration = %f",std::chrono::duration<double, std::milli>(load_duration).count());

        ET_CHECK_MSG(method_name.ok(), "Failed to load method with name=%s from program=%p", method_name.get().c_str(), program.get());
        ET_LOG(Info, "Running method = %s", method_name.get().c_str());

        auto inputs = ::prepare_input_tensors(*method);
        ET_LOG(Info, "Inputs prepared.");

        // Run the model.
        std::vector<double> durations;
        Error status = ::execute_method(&method.get(), args.iterations, durations);
        ET_CHECK_MSG(status == Error::Ok, "Execution of method %s failed with status 0x%" PRIx32, method_name.get().c_str(), status);
        ET_LOG(Info, "Model executed successfully.");

        double mean = ::calculate_mean(durations);
        ET_LOG(Info, "Inference latency=%.2fms.", mean);

        auto outputs = method_allocator.allocateList<EValue>(method->outputs_size());
        status = method->get_outputs(outputs, method->outputs_size());
        ET_CHECK(status == Error::Ok);

        dump_etdump_gen(etdump_gen.get(), debug_buffer, args);

        return EXIT_SUCCESS;
    }
}
