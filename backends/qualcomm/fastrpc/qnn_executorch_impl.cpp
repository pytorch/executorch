#include <chrono>
#include <fstream>
#include <memory>
#include <unordered_map>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <dlfcn.h>
#include "System/QnnSystemInterface.h"
#include "qnn_executorch.h"

#include "HAP_farf.h"

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

class SimpleWrapper {
 public:
  SimpleWrapper(const char* pte_path) {
    auto loader = FileDataLoader::from(pte_path, 256);
    if (!loader.ok()) {
      FARF(
          RUNTIME_ERROR,
          "FileDataLoader::from() failed: 0x%x",
          (int)loader.error());
      return;
    }
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    auto program = Program::load(loader_.get());
    if (!program.ok()) {
      FARF(RUNTIME_ERROR, "failed to parse model file %s", pte_path);
      return;
    }
    program_ = std::make_unique<Program>(std::move(program.get()));

    auto method_name = program_->get_method_name(0);
    if (!method_name.ok()) {
      FARF(RUNTIME_ERROR, "program has no methods");
      return;
    }
    FARF(RUNTIME_HIGH, "using method %s", *method_name);

    auto method_meta = program_->method_meta(*method_name);
    if (!method_meta.ok()) {
      FARF(
          RUNTIME_ERROR,
          "failed to get method_meta for %s: 0x%x",
          *method_name,
          (unsigned int)method_meta.error());
      return;
    }
    method_meta_ = std::make_unique<MethodMeta>(std::move(method_meta.get()));

    method_allocator_ = std::make_unique<MemoryAllocator>(
        sizeof(method_allocator_pool_), method_allocator_pool_);

    for (size_t id = 0; id < method_meta_->num_memory_planned_buffers(); ++id) {
      size_t buffer_size = static_cast<size_t>(
          method_meta->memory_planned_buffer_size(id).get());
      planned_buffers_.push_back(std::make_unique<uint8_t[]>(buffer_size));
      planned_spans_.push_back({planned_buffers_.back().get(), buffer_size});
    }
    planned_memory_ = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>{planned_spans_.data(), planned_spans_.size()});

    memory_manager_ = std::make_unique<MemoryManager>(
        method_allocator_.get(), planned_memory_.get());

    auto method = program_->load_method(*method_name, memory_manager_.get());
    if (!method.ok()) {
      FARF(
          RUNTIME_ERROR,
          "loading of method %s failed with status 0x%x",
          *method_name,
          (int)method.error());
    }
    method_ = std::make_unique<Method>(std::move(method.get()));

    input_tensors_.resize(method_->inputs_size());
    for (int i = 0; i < input_tensors_.size(); ++i) {
      auto tensor_meta = method_meta_->input_tensor_meta(i);
      input_tensors_[i].resize(padded_size(tensor_meta->nbytes()));
      input_tensor_impls_.emplace_back(TensorImpl(
          tensor_meta->scalar_type(),
          tensor_meta->sizes().size(),
          const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
          align_ptr(input_tensors_[i].data()),
          const_cast<TensorImpl::DimOrderType*>(
              tensor_meta->dim_order().data())));
      Error ret = method_->set_input(Tensor(&input_tensor_impls_.back()), i);
      if (ret != Error::Ok) {
        FARF(RUNTIME_ERROR, "failed to set input tensor: %d", (int)ret);
        return;
      }
    }
    output_tensors_.resize(method_->outputs_size());
    for (int i = 0; i < output_tensors_.size(); ++i) {
      auto tensor_meta = method_meta_->output_tensor_meta(i);
      output_tensors_[i].resize(padded_size(tensor_meta->nbytes()));
      Error ret = method_->set_output_data_ptr(
          align_ptr(output_tensors_[i].data()), tensor_meta->nbytes(), i);
      if (ret != Error::Ok) {
        FARF(RUNTIME_ERROR, "failed to set output tensor: %d", (int)ret);
        return;
      }
    }
  }

  size_t padded_size(size_t sz) {
    size_t new_sz = alignment_ + sz;
    return new_sz;
  }

  void* align_ptr(void* ptr) {
    void* addr = reinterpret_cast<void*>(
        ((size_t)ptr + (alignment_ - 1)) & ~(alignment_ - 1));
    return addr;
  }

  int get_input_size(const int index) {
    if (index < input_tensors_.size()) {
      auto tensor_meta = method_meta_->input_tensor_meta(index);
      return tensor_meta.ok() ? tensor_meta->nbytes() : -1;
    }
    return -1;
  }

  void set_input(int index, const tensor& t) {
    if (padded_size(t.dataLen) > input_tensors_[index].size()) {
      FARF(
          RUNTIME_ERROR,
          "input tensor %d size mismatched: %d vs %d",
          index,
          input_tensors_[index].size(),
          t.dataLen);
      return;
    }
    std::memcpy(align_ptr(input_tensors_[index].data()), t.data, t.dataLen);
  }

  int get_output_size(const int index) {
    if (index < output_tensors_.size()) {
      auto tensor_meta = method_meta_->output_tensor_meta(index);
      return tensor_meta.ok() ? tensor_meta->nbytes() : -1;
    }
    return -1;
  }

  void get_output(int index, tensor& t) {
    if (padded_size(t.dataLen) > output_tensors_[index].size()) {
      FARF(
          RUNTIME_ERROR,
          "output tensor %d size mismatched: %d vs %d",
          index,
          output_tensors_[index].size(),
          t.dataLen);
      return;
    }
    std::memcpy(t.data, align_ptr(output_tensors_[index].data()), t.dataLen);
  }

  void execute() {
    Error status = method_->execute();
    if (status != Error::Ok) {
      FARF(
          RUNTIME_ERROR,
          "Execution of method failed with status 0x%x",
          (int)status);
    }
  }

 private:
  uint8_t method_allocator_pool_[4 * 1024U];
  const size_t alignment_ = 64;
  std::unique_ptr<FileDataLoader> loader_;
  std::unique_ptr<HierarchicalAllocator> planned_memory_;
  std::unique_ptr<MethodMeta> method_meta_;
  std::unique_ptr<MemoryAllocator> method_allocator_;
  std::unique_ptr<MemoryManager> memory_manager_;
  std::unique_ptr<Method> method_;
  std::unique_ptr<Program> program_;
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers_;
  std::vector<Span<uint8_t>> planned_spans_;
  std::vector<std::vector<uint8_t>> input_tensors_;
  std::vector<std::vector<uint8_t>> output_tensors_;
  std::vector<TensorImpl> input_tensor_impls_;
  std::vector<TensorImpl> output_tensor_impls_;
};

std::unordered_map<std::string, std::unique_ptr<SimpleWrapper>>
    g_cached_request;

AEEResult qnn_executorch_open(const char* uri, remote_handle64* h) {
  FARF(RUNTIME_HIGH, __func__);
  executorch::runtime::runtime_init();
  return 0;
}

AEEResult qnn_executorch_close(remote_handle64 h) {
  FARF(RUNTIME_HIGH, __func__);
  g_cached_request.clear();
  return 0;
}

AEEResult qnn_executorch_load(remote_handle64 _h, const char* pte_path) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (!g_cached_request.count(key)) {
    g_cached_request[key] = std::make_unique<SimpleWrapper>(pte_path);
  }
  return 0;
}

AEEResult qnn_executorch_get_input_size(
    remote_handle64 _h,
    const char* pte_path,
    const int index,
    int* nbytes) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  *nbytes = -1;
  if (g_cached_request.count(key)) {
    *nbytes = g_cached_request[key]->get_input_size(index);
  }
  return 0;
}

AEEResult qnn_executorch_set_input(
    remote_handle64 _h,
    const char* pte_path,
    const tensor* tensors,
    int tensorsLen) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (g_cached_request.count(key)) {
    auto& wrapper = g_cached_request[key];
    for (int i = 0; i < tensorsLen; ++i) {
      wrapper->set_input(i, tensors[i]);
    }
  }
  return 0;
}

AEEResult qnn_executorch_execute(remote_handle64 _h, const char* pte_path) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (g_cached_request.count(key)) {
    auto before_exec = std::chrono::high_resolution_clock::now();
    g_cached_request[key]->execute();
    auto after_exec = std::chrono::high_resolution_clock::now();
    double interval_infs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            after_exec - before_exec)
            .count() /
        1000.0;
    FARF(RUNTIME_HIGH, "inferences took %f ms", interval_infs);
  }
  return 0;
}

AEEResult qnn_executorch_get_output_size(
    remote_handle64 _h,
    const char* pte_path,
    const int index,
    int* nbytes) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  *nbytes = -1;
  if (g_cached_request.count(key)) {
    *nbytes = g_cached_request[key]->get_output_size(index);
  }
  return 0;
}

AEEResult qnn_executorch_get_output(
    remote_handle64 _h,
    const char* pte_path,
    tensor* tensors,
    int tensorsLen) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (g_cached_request.count(key)) {
    auto& wrapper = g_cached_request[key];
    for (int i = 0; i < tensorsLen; ++i) {
      wrapper->get_output(i, tensors[i]);
    }
  }
  return 0;
}

AEEResult qnn_executorch_unload(remote_handle64 _h, const char* pte_path) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (g_cached_request.count(key)) {
    g_cached_request.erase(key);
  }
  return 0;
}
