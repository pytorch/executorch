//
// main.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import <chrono>
#import <memory>
#import <numeric>
#import <string>

#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/runtime/executor/method.h>
#import <executorch/runtime/executor/program.h>
#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/platform/runtime.h>
#import <executorch/util/util.h>
#import <executorch/sdk/etdump/etdump_flatcc.h>

#import <coreml_backend/delegate.h>

static inline id check_class(id obj, Class cls) {
    return [obj isKindOfClass:cls] ? obj : nil;
}

#define SAFE_CAST(Object, Type) ((Type *)check_class(Object, [Type class]))

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

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
            NSString *value = SAFE_CAST(params[@"--dump_intermediate_outputs"], NSString);
            if (value.length > 0) {
                dump_intermediate_outputs = value.boolValue;
            }
        }
        {
            NSString *value = SAFE_CAST(params[@"--dump_model_outputs"], NSString);
            if (value.length > 0) {
                dump_model_outputs = value.boolValue;
            }
        }
    }
};

NSString *clean_string(NSString *value) {
    return [value stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
}

NSSet<NSString *> *all_keys() {
    return [NSSet setWithObjects:@"--model_path", @"--iterations", @"--purge_models_cache", @"--etdump_path", @"--debug_buffer_path", @"--debug_buffer_size", @"--dump_intermediate_outputs", @"--dump_model_outputs", nil];
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

class DataLoaderImpl: public DataLoader {
public:
    DataLoaderImpl(const std::string& filePath)
    :data_(read_data(filePath))
    {}

    Result<FreeableBuffer> Load(size_t offset, size_t size) override {
        NSData *subdata = [data_ subdataWithRange:NSMakeRange(offset, size)];
        return FreeableBuffer(subdata.bytes, size, nullptr);
    }

    Result<size_t> size() const override {
        return data_.length;
    }

private:
   NSData *data_;
};

using Buffer = std::vector<uint8_t>;

std::unique_ptr<Program> get_program(NSURL *url) {
    DataLoaderImpl dataLoader(url.path.UTF8String);
    auto program = Program::load(&dataLoader);
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
}

int main(int argc, char * argv[]) {
    @autoreleasepool {
        runtime_init();

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

        auto program = get_program(model_url);
        ET_CHECK_MSG(program != nil, "Failed to load program from path=%s", args.model_path.c_str());

        auto method_name = get_method_name(program.get());
        ET_CHECK_MSG(method_name.ok(), "Failed to get method name from program=%p", program.get());

        auto plannedBuffers = get_planned_buffers(method_name.get(), program.get());
        Buffer method_buffer(kRuntimeMemorySize, 0);
        MemoryAllocator method_allocator(static_cast<int32_t>(method_buffer.size()), method_buffer.data());
        auto spans = to_spans(plannedBuffers.get());
        HierarchicalAllocator planned_allocator(Span<Span<uint8_t>>(reinterpret_cast<Span<uint8_t> *>(spans.data()), spans.size()));
        MemoryManager memory_manager(&method_allocator, &planned_allocator);

        ETDumpGen *etdump_gen = new ETDumpGen();
        Buffer debug_buffer(args.debug_buffer_size, 0);
        if (args.dump_intermediate_outputs) {
            ET_LOG(Info, "Dumping intermediate outputs");
            Span<uint8_t> buffer(debug_buffer.data(), debug_buffer.size());
            etdump_gen->set_debug_buffer(buffer);
            etdump_gen->set_event_tracer_debug_level(
                EventTracerDebugLogLevel::kIntermediateOutputs);
        } else if (args.dump_model_outputs) {
            ET_LOG(Info, "Dumping model outputs");
            Span<uint8_t> buffer(debug_buffer.data(), debug_buffer.size());
            etdump_gen->set_debug_buffer(buffer);
            etdump_gen->set_event_tracer_debug_level(
                EventTracerDebugLogLevel::kProgramOutputs);
        }

        auto load_start_time = std::chrono::steady_clock::now();
        auto method = program->load_method(method_name.get().c_str(), &memory_manager, (EventTracer *)etdump_gen);
        auto load_duration = std::chrono::steady_clock::now() - load_start_time;
        ET_LOG(Info, "Load duration = %f",std::chrono::duration<double, std::milli>(load_duration).count());

        ET_CHECK_MSG(method_name.ok(), "Failed to load method with name=%s from program=%p", method_name.get().c_str(), program.get());
        ET_LOG(Info, "Running method = %s", method_name.get().c_str());

        auto inputs = util::PrepareInputTensors(*method);
        ET_LOG(Info, "Inputs prepared.");

        // Run the model.
        std::vector<double> durations;
        Error status = execute_method(&method.get(), args.iterations, durations);
        ET_CHECK_MSG(status == Error::Ok, "Execution of method %s failed with status 0x%" PRIx32, method_name.get().c_str(), status);
        ET_LOG(Info, "Model executed successfully.");

        double mean = calculate_mean(durations);
        ET_LOG(Info, "Inference latency=%.2fms.", mean);

        auto outputs = method_allocator.allocateList<EValue>(method->outputs_size());
        status = method->get_outputs(outputs, method->outputs_size());
        ET_CHECK(status == Error::Ok);

        etdump_result result = etdump_gen->get_etdump_data();
        if (result.size != 0) {
            ET_LOG(Info, "Size = %zu", result.size);
            FILE *ptr = fopen(args.etdump_path.c_str(), "wb");
            fwrite(result.buf, 1, result.size, ptr);
            fclose(ptr);
            if (args.dump_intermediate_outputs || args.dump_model_outputs) {
                FILE *ptr = fopen(args.debug_buffer_path.c_str(), "wb");
                fwrite(debug_buffer.data(), 1, debug_buffer.size(), ptr);
                fclose(ptr);
            }
        }

        util::FreeInputs(inputs);
        return 0;
    }
}
