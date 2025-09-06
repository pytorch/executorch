/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "ModelChunk.h"

#include <sstream>

#include "executorch/backends/mediatek/runtime/include/NeuronBufferAllocator.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#define ENSURE_INIT \
  ET_CHECK_MSG(Initialized(), "Error: Model chunk not initialized.");

namespace example {

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
using executorch::runtime::Tag;

static constexpr size_t kMethodAllocatorPoolSize = 4 * 1024U * 1024U; // 4MB

// ExecuTorch model instance with cacheable program.
class ModelInstance {
 public:
  explicit ModelInstance(const std::string& modelPath) {
    if (mCachedPrograms.find(modelPath) != mCachedPrograms.end()) {
      auto cachedProgram = mCachedPrograms.at(modelPath).lock();
      if (cachedProgram) {
        mProgramInstance = cachedProgram;
        ET_LOG(
            Debug, "Loaded existing program from cache: %s", modelPath.c_str());
        return;
      } else {
        mCachedPrograms.erase(modelPath); // Expired
      }
    }
    ET_LOG(Debug, "Loading model from scratch: %s", modelPath.c_str());
    mProgramInstance = std::make_shared<ProgramInstance>();

    // Create a loader to get the data of the program file. There are other
    // DataLoaders that use mmap() or point to data that's already in memory,
    // and users can create their own DataLoaders to load from arbitrary
    // sources.
    Result<FileDataLoader> loader = FileDataLoader::from(modelPath.c_str());
    ET_CHECK_MSG(
        loader.ok(),
        "FileDataLoader::from() failed: 0x%" PRIx32,
        loader.error());
    // Extract the data loader out to a persistent storage before loading the
    // program.
    mProgramInstance->dataLoader =
        std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Parse the program file. This is immutable, and can also be reused between
    // multiple execution invocations across multiple threads.
    Result<Program> program_loaded =
        Program::load(mProgramInstance->dataLoader.get());
    ET_CHECK_MSG(
        program_loaded.ok(),
        "Failed to parse model file %s",
        modelPath.c_str());
    ET_LOG(Debug, "Model file %s is loaded.", modelPath.c_str());

    // Extract program out to a persistent storage before calling any of its
    // methods.
    mProgramInstance->program =
        std::make_unique<Program>(std::move(program_loaded.get()));
    mCachedPrograms.emplace(modelPath, mProgramInstance);
  }

  Method& GetMethod() {
    ET_CHECK_MSG(mMethod != nullptr, "Method is not loaded.");
    return *mMethod;
  }

  const Method& GetMethod() const {
    ET_CHECK_MSG(mMethod != nullptr, "Method is not loaded.");
    return *mMethod;
  }

  Program& GetProgram() {
    return *(mProgramInstance->program);
  }

  const Program& GetProgram() const {
    return *(mProgramInstance->program);
  }

  std::vector<std::string> GetMethodNames() const {
    std::vector<std::string> methodNames;
    for (size_t i = 0; i < GetProgram().num_methods(); i++) {
      const auto method_name_result = GetProgram().get_method_name(i);
      ET_CHECK_MSG(method_name_result.ok(), "Program has no method %zu", i);
      methodNames.emplace_back(*method_name_result);
    }
    return methodNames;
  }

  void LoadFirstMethod() {
    // Use the first method in the program.
    const char* method_name = nullptr;
    const auto method_name_result = GetProgram().get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
    ET_LOG(Debug, "Loading the first method.");
    LoadMethod(method_name);
  }

  void LoadMethod(const std::string& method_name) {
    ET_CHECK_MSG(!mMethod, "Method is already loaded.");
    const auto method_name_cstr = method_name.c_str();

    // MethodMeta describes the memory requirements of the method.
    Result<MethodMeta> method_meta = GetProgram().method_meta(method_name_cstr);
    ET_CHECK_MSG(
        method_meta.ok(),
        "Failed to get method_meta for %s: 0x%x",
        method_name_cstr,
        (unsigned int)method_meta.error());

    mMethodAllocatorPool.resize(kMethodAllocatorPoolSize);
    mMethodAllocator = std::make_unique<MemoryAllocator>(
        kMethodAllocatorPoolSize, mMethodAllocatorPool.data());
    mMethodAllocator->enable_profiling("method allocator");

    size_t num_memory_planned_buffers =
        method_meta->num_memory_planned_buffers();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      // .get() will always succeed because id < num_memory_planned_buffers.
      size_t buffer_size = static_cast<size_t>(
          method_meta->memory_planned_buffer_size(id).get());
      ET_LOG(
          Debug, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
      mPlannedBuffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
      mPlannedSpans.push_back({mPlannedBuffers.back().get(), buffer_size});
    }
    mPlannedMemory = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>{mPlannedSpans.data(), mPlannedSpans.size()});

    // Assemble all of the allocators into the MemoryManager that the Executor
    // will use.
    auto& neuron_allocator = GET_NEURON_ALLOCATOR;
    mMemoryManager = std::make_unique<MemoryManager>(
        mMethodAllocator.get(),
        mPlannedMemory.get(),
        dynamic_cast<MemoryAllocator*>(&neuron_allocator));

    ET_LOG(Debug, "Loading method %s", method_name_cstr);
    Result<Method> method =
        GetProgram().load_method(method_name_cstr, mMemoryManager.get());
    ET_CHECK_MSG(
        method.ok(),
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name_cstr,
        method.error());

    mMethod = std::make_unique<Method>(std::move(method.get()));
  }

 private:
  ModelInstance(const ModelInstance&) = delete;
  ModelInstance& operator=(ModelInstance&&) = delete;
  ModelInstance& operator=(const ModelInstance&) = delete;

 private:
  // The member ordering below affects the order of destruction.

  struct ProgramInstance {
    std::unique_ptr<FileDataLoader> dataLoader;
    std::unique_ptr<Program> program;
  };
  std::shared_ptr<ProgramInstance> mProgramInstance;

  std::vector<std::unique_ptr<uint8_t[]>> mPlannedBuffers;
  std::vector<Span<uint8_t>> mPlannedSpans;

  std::vector<uint8_t> mMethodAllocatorPool;
  std::unique_ptr<MemoryAllocator> mMethodAllocator;
  std::unique_ptr<HierarchicalAllocator> mPlannedMemory;
  std::unique_ptr<MemoryManager> mMemoryManager;

  std::unique_ptr<Method> mMethod;

  // Maps .pte file paths to the cached program instances.
  inline static std::unordered_map<std::string, std::weak_ptr<ProgramInstance>>
      mCachedPrograms;
};

void ModelChunk::Initialize() {
  LoadModels();
  GetModelIoInfo();
  AllocateIoBuffers();
  SetBackendInputs();
  SetBackendOutputs();
  mIsInitialized = true;
}

bool ModelChunk::Initialized() {
  return mIsInitialized;
}

void ModelChunk::Release() {
  ENSURE_INIT
  ReleaseModels();
  ReleaseIoBuffers();
}

void ModelChunk::Run() {
  ENSURE_INIT
  auto beforeExec = std::chrono::high_resolution_clock::now();
  Error status = Error::Ok;
  status = GetModelMethod().execute();
  auto afterExec = std::chrono::high_resolution_clock::now();
  const double elapsedTime =
      std::chrono::duration_cast<std::chrono::microseconds>(
          afterExec - beforeExec)
          .count();
  ET_LOG(Debug, "Inference took %f ms", elapsedTime / 1000.0);
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method failed with status 0x%" PRIx32,
      status);
  ET_LOG(Debug, "Model executed successfully.");
}

bool ModelChunk::HotSwapModel(const size_t tokenBatchSize) {
  ENSURE_INIT
  // Save old values
  const auto oldInstanceBatchSize = GetModelId();
  const auto oldTokenBatchSize = mTokenBatchSize;

  if (!HasModel(tokenBatchSize)) {
    ET_LOG(
        Error,
        "Model swap: No model with batchSize=%zu is available",
        tokenBatchSize);
    return false;
  }

  if (oldInstanceBatchSize == tokenBatchSize) {
    ET_LOG(Info, "Model swapping to itself");
    return true;
  }

  SelectModel(tokenBatchSize);

  const auto newInstanceBatchSize = GetModelId();
  if (oldInstanceBatchSize == newInstanceBatchSize) {
    ET_LOG(
        Error,
        "Failed to switch to model with batchSize=%zu. Model currently remain at batchSize=%zu",
        tokenBatchSize,
        oldTokenBatchSize);
    return false;
  }

  // Update model variables
  // Mask length = cache size (length) + num input token (token batch size)
  mTokenBatchSize = tokenBatchSize;

  UpdateModelIoInfo();
  SetBackendInputs();
  SetBackendOutputs();
  return true;
}

void ModelChunk::SetInputBuffer(
    const void* data,
    const size_t size,
    const size_t index) {
  ENSURE_INIT
  auto& targetBufInfo = mInputBufferInfos[index];
  ET_CHECK_MSG(
      targetBufInfo.nbytes >= size,
      "Error: Input[%zu] has only allocated %zu but need to set input with size %zu",
      index,
      targetBufInfo.nbytes,
      size);
  std::memcpy(targetBufInfo.data, data, size);
}

void ModelChunk::SetInputBuffer(
    const BufferInfo& bufferInfo,
    const size_t index) {
  // Allow calling this method without initialized first to assign preallocated
  // buffers.
  if (index >= mInputBufferInfos.size()) {
    mInputBufferInfos.resize(index + 1);
  }
  // If the existing buffer has been allocated, memory copy the content.
  // Otherwise, share the input buffer info.
  auto& targetBufInfo = mInputBufferInfos[index];
  if (targetBufInfo.data != nullptr) {
    // Already allocated, do memcpy.
    SetInputBuffer(bufferInfo.data, bufferInfo.nbytesUsed, index);
  } else {
    // Share the buffer info.
    targetBufInfo = bufferInfo;
  }
}

BufferInfo ModelChunk::GetInputBuffer(const size_t index) {
  ENSURE_INIT
  ET_CHECK_MSG(
      index < mInputBufferInfos.size(),
      "Error: Index out of range: %zu",
      index);
  return mInputBufferInfos[index];
}

BufferInfo ModelChunk::GetOutputBuffer(const size_t index) {
  ENSURE_INIT
  ET_CHECK_MSG(
      index < mOutputBufferInfos.size(),
      "Error: Index out of range: %zu",
      index);
  return mOutputBufferInfos[index];
}

void ModelChunk::LogIoSummary() {
  ENSURE_INIT
  const auto& method = GetModelMethod();
  const auto method_meta = method.method_meta();

  auto getShapeStr = [](const auto shape) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); i++) {
      ss << shape[i];
      if (i < shape.size() - 1)
        ss << ", ";
    }
    ss << ")";
    return ss.str();
  };

  ET_LOG(Info, "Model Chunk IO Summary:");

  const size_t input_size = method.inputs_size();
  const size_t output_size = method.outputs_size();

  for (size_t i = 0; i < input_size; i++) {
    if (*method_meta.input_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "  Input %zu: Non-Tensor", i);
      continue;
    }
    const auto nbytes = method_meta.input_tensor_meta(i)->nbytes();
    const auto shape = getShapeStr(method_meta.input_tensor_meta(i)->sizes());
    const auto type =
        static_cast<int>(method_meta.input_tensor_meta(i)->scalar_type());
    ET_LOG(
        Info,
        "  Input %zu: Shape: %s, Size: %zu bytes, Type: %d",
        i,
        shape.c_str(),
        nbytes,
        type);
  }

  for (size_t i = 0; i < output_size; i++) {
    if (*method_meta.output_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "  Output %zu: Non-Tensor", i);
      continue;
    }
    const auto nbytes = method_meta.output_tensor_meta(i)->nbytes();
    const auto shape = getShapeStr(method_meta.output_tensor_meta(i)->sizes());
    const auto type =
        static_cast<int>(method_meta.output_tensor_meta(i)->scalar_type());
    ET_LOG(
        Info,
        "  Output %zu: Shape: %s, Size: %zu bytes, Type: %d",
        i,
        shape.c_str(),
        nbytes,
        type);
  }
}

void ModelChunk::GetModelIoInfo() {
  const auto& method = GetModelMethod();
  const auto method_meta = method.method_meta();

  const size_t input_size = method.inputs_size();
  const size_t output_size = method.outputs_size();

  mInputBufferInfos.resize(input_size);
  for (size_t i = 0; i < input_size; i++) {
    if (*method_meta.input_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "Input %zu is not a tensor, skipping", i);
      continue;
    }
    auto& bufInfo = mInputBufferInfos[i];
    const auto nbytes = method_meta.input_tensor_meta(i)->nbytes();
    if (bufInfo.data != nullptr) {
      // Already preallocated, so just update the size used by the model.
      ET_CHECK_MSG(
          bufInfo.nbytes >= nbytes,
          "Error: Model input[%zu] requires size=%zu but only preallocated size=%zu",
          i,
          nbytes,
          bufInfo.nbytes);
      bufInfo.nbytesUsed = nbytes;
      continue;
    }
    bufInfo.nbytes = nbytes;
    bufInfo.nbytesUsed = nbytes;
  }

  mOutputBufferInfos.resize(output_size);
  for (size_t i = 0; i < output_size; i++) {
    if (*method_meta.output_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "Output %zu is not a tensor, skipping", i);
      continue;
    }
    auto& bufInfo = mOutputBufferInfos[i];
    const auto nbytes = method_meta.output_tensor_meta(i)->nbytes();
    if (bufInfo.data != nullptr) {
      // Already preallocated, so just update the size used by model.
      ET_CHECK_MSG(
          bufInfo.nbytes >= nbytes,
          "Error: Model output[%zu] requires size of %zu but only preallocated size of %zu",
          i,
          nbytes,
          bufInfo.nbytes);
      bufInfo.nbytesUsed = nbytes;
      continue;
    }
    bufInfo.nbytes = nbytes;
    bufInfo.nbytesUsed = nbytes;
  }
}

// Update actual used IO sizes by the model
void ModelChunk::UpdateModelIoInfo() {
  const auto& method = GetModelMethod();
  const auto method_meta = method.method_meta();

  const size_t numModelInputs = method.inputs_size();
  const size_t numModelOutputs = method.outputs_size();

  const size_t numInputBuffers = mInputBufferInfos.size();
  const size_t numOutputBuffers = mOutputBufferInfos.size();

  if (numInputBuffers != numModelInputs) {
    ET_LOG(
        Info,
        "Existing num inputs (%zu) != new num inputs (%zu)",
        numInputBuffers,
        numModelInputs);
  }
  if (numOutputBuffers != numModelOutputs) {
    ET_LOG(
        Info,
        "Existing num outputs (%zu) != new num outputs (%zu)",
        numOutputBuffers,
        numModelOutputs);
  }
  mInputBufferInfos.resize(numModelInputs);
  for (size_t inputIdx = 0; inputIdx < numModelInputs; inputIdx++) {
    auto& sizeAllocated = mInputBufferInfos[inputIdx].nbytes;
    auto& sizeRequired = mInputBufferInfos[inputIdx].nbytesUsed;
    const auto before = sizeRequired;

    // Update
    sizeRequired = method_meta.input_tensor_meta(inputIdx)->nbytes();
    if (sizeAllocated < sizeRequired) {
      ET_LOG(
          Error,
          "Insufficient buffer size for input[%zu]. Requires %zu but only allocated %zu",
          inputIdx,
          sizeRequired,
          sizeAllocated);
    }
    if (before != sizeRequired) {
      ET_LOG(
          Debug,
          "Update input[%zu] size:  %zu -> %zu",
          inputIdx,
          before,
          sizeRequired);
    }
  }
  mOutputBufferInfos.resize(numModelOutputs);
  for (size_t outputIdx = 0; outputIdx < numModelOutputs; outputIdx++) {
    auto& sizeAllocated = mOutputBufferInfos[outputIdx].nbytes;
    auto& sizeRequired = mOutputBufferInfos[outputIdx].nbytesUsed;
    const auto before = sizeRequired;

    // Update
    sizeRequired = method_meta.output_tensor_meta(outputIdx)->nbytes();
    if (sizeAllocated < sizeRequired) {
      ET_LOG(
          Error,
          "Insufficient buffer size for output[%zu]. Requires %zu but only allocated %zu",
          outputIdx,
          sizeRequired,
          sizeAllocated);
    }
    if (before != sizeRequired) {
      ET_LOG(
          Debug,
          "Update output[%zu] size:  %zu -> %zu",
          outputIdx,
          before,
          sizeRequired);
    }
  }
}

void ModelChunk::LinkModelIO(
    const size_t inputIndex,
    const size_t outputIndex) {
  mModelOutToInIndexLinks.emplace(outputIndex, inputIndex);
}

std::optional<size_t> ModelChunk::GetLinkedInputIndex(
    const size_t outputIndex) const {
  auto hasKey = [](const auto& map, const auto& key) {
    return map.find(key) != map.end();
  };
  if (hasKey(mModelOutToInIndexLinks, outputIndex))
    return mModelOutToInIndexLinks.at(outputIndex);
  else
    return std::nullopt;
}

void ModelChunk::SetBackendInputs() {
  auto& method = GetModelMethod();
  const auto method_meta = method.method_meta();
  const size_t input_size = method.inputs_size();
  for (size_t i = 0; i < input_size; i++) {
    const auto tensor_meta = method_meta.input_tensor_meta(i);
    auto scalar_type = tensor_meta->scalar_type();
    auto sizes_raw = tensor_meta->sizes();
    auto dim = sizes_raw.size();
    auto dim_order_raw = tensor_meta->dim_order();
    std::vector sizes(sizes_raw.begin(), sizes_raw.end());
    std::vector dim_order(dim_order_raw.begin(), dim_order_raw.end());
    auto buffer_data = mInputBufferInfos[i].data;

    TensorImpl impl = TensorImpl(
        scalar_type, dim, sizes.data(), buffer_data, dim_order.data());
    Tensor tensor(&impl);
    const auto error = method.set_input(tensor, i);
    ET_CHECK_MSG(
        error == Error::Ok,
        "Error: 0x%" PRIx32 " setting input %zu.",
        error,
        i);
  }
}

void ModelChunk::SetBackendOutputs() {
  auto& method = GetModelMethod();
  for (size_t i = 0; i < mOutputBufferInfos.size(); i++) {
    auto data = mOutputBufferInfos[i].data;
    const auto nbytes = mOutputBufferInfos[i].nbytes;
    const auto output_err = method.set_output_data_ptr(data, nbytes, i);
    ET_CHECK_MSG(
        output_err == Error::Ok,
        "Error: 0x%" PRIx32 " setting output %zu.",
        output_err,
        i);
  }
}

void ModelChunk::AllocateIoBuffers() {
  auto& buffer_allocator = GET_NEURON_ALLOCATOR;

  // Inputs
  for (auto& inBufInfo : mInputBufferInfos) {
    if (inBufInfo.data != nullptr) {
      continue; // Already allocated
    }
    void* ahwb_data = buffer_allocator.Allocate(inBufInfo.nbytes);
    inBufInfo.data = ahwb_data;
  }

  // Outputs
  const auto numOutputBuffers = mOutputBufferInfos.size();
  for (size_t outputIdx = 0; outputIdx < numOutputBuffers; outputIdx++) {
    auto& outBufInfo = mOutputBufferInfos[outputIdx];
    if (outBufInfo.data != nullptr) {
      continue; // Already allocated
    }
    const auto linkedInputIdx = GetLinkedInputIndex(outputIdx);
    if (linkedInputIdx) {
      const auto& linkedInBufInfo = mInputBufferInfos[*linkedInputIdx];
      // Ensure the linked IO sizes match, then reuse the linked input buffer
      ET_CHECK_MSG(
          outBufInfo.nbytes == linkedInBufInfo.nbytes,
          "Error: Mismatch sizes between linked IO. "
          "Input %zu size is %zu, but Output %zu size is %zu.",
          *linkedInputIdx,
          linkedInBufInfo.nbytes,
          outputIdx,
          outBufInfo.nbytes);
      outBufInfo = linkedInBufInfo;
      continue;
    }
    // Allocate output buffer as usual
    void* ahwb_data = buffer_allocator.Allocate(outBufInfo.nbytes);
    outBufInfo.data = ahwb_data;
  }
}

void ModelChunk::ReleaseIoBuffers() {
  auto& buffer_allocator = GET_NEURON_ALLOCATOR;

  for (size_t i = 0; i < mInputBufferInfos.size(); i++)
    buffer_allocator.RemoveBuffer(mInputBufferInfos[i].data);

  for (size_t i = 0; i < mOutputBufferInfos.size(); i++)
    buffer_allocator.RemoveBuffer(mOutputBufferInfos[i].data);
}

Method& ModelChunk::GetModelMethod() {
  auto modelInstance = reinterpret_cast<ModelInstance*>(GetModelInstance());
  return modelInstance->GetMethod();
}

// Override the virtual functions
void* ModelChunk::CreateModelInstance(const std::string& modelPath) {
  auto modelInstance = new ModelInstance(modelPath);
  const auto selectedMethod = SelectMethod(modelInstance->GetMethodNames());
  if (!selectedMethod.empty()) {
    modelInstance->LoadMethod(selectedMethod);
  } else {
    modelInstance->LoadFirstMethod(); // Load the first available method
  }
  return modelInstance;
}

void ModelChunk::ReleaseModelInstance(void* modelInstance) {
  if (modelInstance != nullptr) {
    delete reinterpret_cast<ModelInstance*>(modelInstance);
  }
}

} // namespace example
