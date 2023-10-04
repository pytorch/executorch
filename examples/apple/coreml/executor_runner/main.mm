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

#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/runtime/executor/method.h>
#import <executorch/runtime/executor/program.h>
#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/platform/profiler.h>
#import <executorch/runtime/platform/runtime.h>
#import <executorch/util/util.h>

#import <coreml/coreml_backend_delegate.h>

#import <gflags/gflags.h>

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

static constexpr size_t kRuntimeMemorySize = 16 * 1024U * 1024U; // 16 MB

DECLARE_int32(iterations);

DEFINE_string(model_path, "model.pte", "Model serialized in pte format.");
DEFINE_int32(iterations, 1, "Number of iterations.");
DEFINE_int32(purge_models_cache, 0, "Purges the compiled models cache.");
DEFINE_string(prof_result_path, "prof_result.bin", "Executorch profiler output path.");

namespace {

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

void dump_profile_data() {
    // Dump the profiling data to the specified file.
    torch::executor::prof_result_t prof_result;
    EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
    if (prof_result.num_bytes != 0) {
      FILE* ptr = fopen(FLAGS_prof_result_path.c_str(), "w+");
      fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
      fclose(ptr);
    }
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
    runtime_init();
    
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_purge_models_cache > 0) {
        ET_LOG(Info, "Purging models cache");
        auto delegate = CoreMLBackendDelegate::get_registered_delegate();
        delegate->purge_models_cache();
    }

    const char* model_path = FLAGS_model_path.c_str();
    NSURL *model_url = [NSURL fileURLWithPath:@(model_path)];
    auto program = get_program(model_url);
    ET_CHECK_MSG(program != nil, "Failed to load program from path=%s", model_path);
    
    auto method_name = get_method_name(program.get());
    ET_CHECK_MSG(method_name.ok(), "Failed to get method name from program=%p", program.get());
    
    auto plannedBuffers = get_planned_buffers(method_name.get(), program.get());
    Buffer method_buffer(kRuntimeMemorySize, 0);
    MemoryAllocator method_allocator(static_cast<int32_t>(method_buffer.size()), method_buffer.data());
    auto spans = to_spans(plannedBuffers.get());
    HierarchicalAllocator planned_allocator(Span<Span<uint8_t>>(reinterpret_cast<Span<uint8_t> *>(spans.data()), spans.size()));
    MemoryManager memory_manager(&method_allocator, &planned_allocator);
    
    auto load_start_time = std::chrono::steady_clock::now();
    auto method = program->load_method(method_name.get().c_str(), &memory_manager);
    auto load_duration = std::chrono::steady_clock::now() - load_start_time;
    ET_LOG(Info, "Load duration = %f",std::chrono::duration<double, std::milli>(load_duration).count());
    
    ET_CHECK_MSG(method_name.ok(), "Failed to load method with name=%s from program=%p", method_name.get().c_str(), program.get());
    ET_LOG(Info, "Running method = %s", method_name.get().c_str());

    auto inputs = util::PrepareInputTensors(*method);
    ET_LOG(Info, "Inputs prepared.");

    // Run the model.
    std::vector<double> durations;
    Error status = execute_method(&method.get(), static_cast<size_t>(FLAGS_iterations), durations);
    ET_CHECK_MSG(status == Error::Ok, "Execution of method %s failed with status 0x%" PRIx32, method_name.get().c_str(), status);
    ET_LOG(Info, "Model executed successfully.");

    double mean = calculate_mean(durations);
    ET_LOG(Info, "Inference latency=%.2fms.", mean);

    auto outputs = method_allocator.allocateList<EValue>(method->outputs_size());
    status = method->get_outputs(outputs, method->outputs_size());
    ET_CHECK(status == Error::Ok);
    
    dump_profile_data();
    util::FreeInputs(inputs);
    return 0;
}
