// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/extension/threadpool/threadpool.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;
// Must match CPU_DELEGATE_VERSION in partitioner.py.
constexpr uint32_t kCpuDelegateVersion = 1;

std::optional<RuntimeConfigurationFactory>& runtime_configuration_factory() {
  static std::optional<RuntimeConfigurationFactory> factory;
  return factory;
}

ExecutionContext execution_context(runtime::EventTracer* tracer = nullptr) {
  ExecutionContext context;
  context.threadpool = extension::threadpool::get_pthreadpool();
  context.threads = context.threadpool
      ? extension::threadpool::get_threadpool()->get_thread_count()
      : 1;
#if defined(__x86_64__) || defined(__i386__)
  context.avx2_fma =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
  context.event_tracer = tracer;
  return context;
}

class CPUExecution {
 public:
  explicit CPUExecution(ptn::Program program) : program_(std::move(program)) {}
  ~CPUExecution() = default;
  CPUExecution(const CPUExecution&) = delete;
  CPUExecution& operator=(const CPUExecution&) = delete;
  CPUExecution(CPUExecution&&) = delete;
  CPUExecution& operator=(CPUExecution&&) = delete;

  Error initialize(BackendInitContext& context) {
    const auto start = std::chrono::steady_clock::now();
    method_ = &program_.get_method("forward");
    auto& allocator = *context.get_runtime_allocator();
    const auto& graph = method_->graph;
    ET_CHECK_OR_RETURN_ERROR(
        graph.subgraphs.empty(),
        NotSupported,
        "CpuBackend does not support control flow");
    for (const auto& spec : method_->output_specs) {
      ET_CHECK_OR_RETURN_ERROR(
          spec.kind == ptn::OutputKind::UserOutput,
          NotSupported,
          "CpuBackend static FP32 delegate does not support mutation");
    }
    buffers_.resize(graph.values.size());
    bytes_.reserve(graph.values.size());
    for (const auto& value : graph.values) {
      ET_CHECK_OR_RETURN_ERROR(
          value.alias_id == ptn::kInvalid,
          NotSupported,
          "CPU static delegate requires unaliased values: %s",
          value.name.c_str());
      auto bytes = tensor_bytes(value);
      if (!bytes.ok()) {
        return bytes.error();
      }
      bytes_.push_back(bytes.get());
    }
    const auto factory = runtime_configuration_factory();
    ET_CHECK_OR_RETURN_ERROR(
        factory.has_value(),
        InvalidState,
        "CpuBackend requires a cpu_backend() provider composition");
    auto configuration = (*factory)();
    auto error = configuration.apply_options(context);
    if (error != Error::Ok) {
      return error;
    }
    plan_ = std::make_unique<CPUPlan>(
        graph, buffers_, std::move(configuration), execution_context());
    error = plan_->select();
    if (error != Error::Ok) {
      return error;
    }
    input_ids_.reserve(graph.input_ids.size());
    for (auto id : graph.input_ids) {
      if (graph.value(id).role == ptn::ValueRole::UserInput) {
        input_ids_.push_back(id);
      }
    }
    constants_.reserve(method_->data_bindings.size());
    const auto* named_data = context.get_named_data_map();
    for (const auto& binding : method_->data_bindings) {
      ET_CHECK_OR_RETURN_ERROR(
          named_data && binding.has_data && !binding.mutated,
          NotSupported,
          "CpuBackend requires immutable constants: %s",
          binding.key.c_str());
      auto data = named_data->get_data(binding.key);
      if (!data.ok()) {
        return data.error();
      }
      const auto id = binding.value_id;
      const auto bytes = bytes_.at(id);
      ET_CHECK_OR_RETURN_ERROR(
          data->size() >= bytes,
          InvalidProgram,
          "CPU constant logical span exceeds storage: %s",
          binding.key.c_str());
      Buffer buffer{
          const_cast<void*>(data->data()), data->size(), 0, kBufferAlignment};
      if (!buffer.accepts(plan_->requirements().at(id))) {
        auto copy = allocate_buffer(allocator, plan_->requirements().at(id));
        if (!copy.ok()) {
          return copy.error();
        }
        if (bytes) {
          std::memcpy(copy->data, data->data(), bytes);
        }
        buffer = copy.get();
        buffer.writable_bytes = 0;
        buffer.owner = StorageOwner::ConstantCopy;
        copied_constant_bytes_ += buffer.readable_bytes;
      }
      buffers_[id] = buffer;
      constants_.push_back(std::move(data.get()));
    }
    error = plan_->prepare(allocator);
    if (error != Error::Ok) {
      return error;
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
    trace(milliseconds);
    return Error::Ok;
  }

  Error execute(BackendExecutionContext& context, Span<EValue*> args) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& outputs = method_->graph.output_ids;
    ET_CHECK_OR_RETURN_ERROR(
        args.size() == input_ids_.size() + outputs.size(),
        InvalidArgument,
        "CPU delegate input/output count mismatch");
    for (size_t index = 0; index < args.size(); ++index) {
      const auto id = index < input_ids_.size()
          ? input_ids_.at(index)
          : outputs[index - input_ids_.size()];
      ET_CHECK_OR_RETURN_ERROR(
          args[index]->isTensor(),
          InvalidArgument,
          "CPU binding must be a tensor");
      const auto& tensor = args[index]->toTensor();
      const auto& expected = method_->graph.value(id).tensor_meta();
      ET_CHECK_OR_RETURN_ERROR(
          tensor.scalar_type() == aten::ScalarType::Float &&
              tensor.dim() == expected.sizes.size() &&
              std::equal(
                  expected.sizes.begin(),
                  expected.sizes.end(),
                  tensor.sizes().begin()) &&
              tensor.nbytes() == bytes_[id] &&
              (tensor.nbytes() == 0 || tensor.const_data_ptr()),
          InvalidArgument,
          "CPU static shape/dtype/layout/storage mismatch: %s",
          method_->graph.value(id).name.c_str());
      int64_t stride = 1;
      for (size_t reverse = expected.sizes.size(); reverse > 0; --reverse) {
        const size_t dim = reverse - 1;
        ET_CHECK_OR_RETURN_ERROR(
            tensor.dim_order()[dim] == dim && tensor.strides()[dim] == stride,
            InvalidArgument,
            "CPU binding must be contiguous: %s",
            method_->graph.value(id).name.c_str());
        stride *= expected.sizes[dim];
      }
    }
    // EValue exposes the logical span, but no backing capacity or readable
    // tail.
    for (size_t index = 0; index < input_ids_.size(); ++index) {
      const auto id = input_ids_[index];
      if (bytes_.at(id)) {
        std::memcpy(
            buffers_.at(id).data,
            args[index]->toTensor().const_data_ptr(),
            bytes_.at(id));
      }
    }
    const auto error =
        plan_->execute(execution_context(context.event_tracer()));
    if (error != Error::Ok) {
      return error;
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      const auto id = outputs[index];
      if (bytes_.at(id)) {
        std::memcpy(
            args[input_ids_.size() + index]->toTensor().mutable_data_ptr(),
            buffers_.at(id).data,
            bytes_.at(id));
      }
    }
    return Error::Ok;
  }

 private:
  void trace(double preparation_ms) const {
    std::string report = plan_->describe(preparation_ms);
    report.pop_back();
    report +=
        ",\"copied_constant_bytes\":" + std::to_string(copied_constant_bytes_) +
        "}";
    constexpr size_t chunk_bytes = 80;
    constexpr std::string_view hex = "0123456789abcdef";
    const size_t chunks = (report.size() + chunk_bytes - 1) / chunk_bytes;
    for (size_t index = 0; index < chunks; ++index) {
      std::string encoded;
      encoded.reserve(chunk_bytes * 2);
      const size_t end = std::min(report.size(), (index + 1) * chunk_bytes);
      for (size_t offset = index * chunk_bytes; offset < end; ++offset) {
        const auto byte = static_cast<unsigned char>(report[offset]);
        encoded += hex.at(byte >> 4);
        encoded += hex.at(byte & 15);
      }
      ET_LOG(Info, "CPU_PLAN_CHUNK %zu %zu %s", index, chunks, encoded.c_str());
    }
  }

  ptn::Program program_;
  const ptn::Method* method_ = nullptr;
  std::vector<FreeableBuffer> constants_;
  std::vector<size_t> bytes_;
  std::vector<Buffer> buffers_;
  std::vector<ptn::ValueId> input_ids_;
  std::unique_ptr<CPUPlan> plan_;
  size_t copied_constant_bytes_ = 0;
  std::mutex mutex_;
};

struct DestroyExecution {
  void operator()(CPUExecution* execution) const {
    execution->~CPUExecution();
  }
};

class CpuBackend final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ET_CHECK_OR_RETURN_ERROR(
        compile_specs.size() == 1 && compile_specs[0].key &&
            compile_specs[0].value.buffer &&
            std::strcmp(compile_specs[0].key, "cpu_delegate_version") == 0 &&
            compile_specs[0].value.nbytes == sizeof(uint32_t),
        NotSupported,
        "CpuBackend requires a uint32 little-endian delegate version");
    const auto* data =
        static_cast<const uint8_t*>(compile_specs[0].value.buffer);
    const uint32_t version = static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);
    ET_CHECK_OR_RETURN_ERROR(
        version == kCpuDelegateVersion,
        NotSupported,
        "Unsupported CpuBackend delegate version %u (expected %u)",
        version,
        kCpuDelegateVersion);
    try {
      auto program = ptn::Program::load(processed->data(), processed->size());
      auto* storage =
          context.get_runtime_allocator()->allocateInstance<CPUExecution>();
      ET_CHECK_OR_RETURN_ERROR(
          storage, MemoryAllocationFailed, "CPU state allocation failed");
      std::unique_ptr<CPUExecution, DestroyExecution> execution(
          new (storage) CPUExecution(std::move(program)));
      const auto error = execution->initialize(context);
      if (error != Error::Ok) {
        return error;
      }
      processed->Free();
      return execution.release();
    } catch (const std::bad_alloc&) {
      return Error::MemoryAllocationFailed;
    } catch (const std::exception& error) {
      ET_LOG(Error, "Invalid CPU delegate: %s", error.what());
      return Error::InvalidProgram;
    }
  }
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    return static_cast<CPUExecution*>(handle)->execute(context, args);
  }
  void destroy(DelegateHandle* handle) const override {
    static_cast<CPUExecution*>(handle)->~CPUExecution();
  }
};

// Backend stores a non-const pointer.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
CpuBackend backend;
const Backend registration{"CpuBackend", &backend};
const auto registered = register_backend(registration);
} // namespace

// Called by the provider registry generated in cpu_backend.bzl.
// cppcheck-suppress unusedFunction
void register_runtime_configuration(RuntimeConfigurationFactory factory) {
  runtime_configuration_factory() = factory;
}
} // namespace executorch::backends::cpu
