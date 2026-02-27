/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/nvidia/tensorrt/runtime/TensorRTExecutor.h>

#include <cstring>

#include <NvInferRuntime.h>

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace tensorrt {

using executorch::runtime::Error;

namespace {

class TensorRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        ET_LOG(Error, "TensorRT: %s", msg);
        break;
      case Severity::kWARNING:
        ET_LOG(Info, "TensorRT: %s", msg);
        break;
      case Severity::kINFO:
        ET_LOG(Info, "TensorRT: %s", msg);
        break;
      default:
        break;
    }
  }
};

TensorRTLogger& get_logger() {
  static TensorRTLogger logger;
  return logger;
}

size_t get_dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      return 4;
  }
}

} // namespace

TensorRTExecutor::~TensorRTExecutor() {
  free_gpu_buffers();
  if (stream_ && owns_stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

TensorRTExecutor::TensorRTExecutor(TensorRTExecutor&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      engine_(std::move(other.engine_)),
      context_(std::move(other.context_)),
      stream_(other.stream_),
      owns_stream_(other.owns_stream_),
      io_bindings_(std::move(other.io_bindings_)),
      gpu_buffers_(std::move(other.gpu_buffers_)),
      uses_unified_memory_(other.uses_unified_memory_) {
  other.stream_ = nullptr;
  other.owns_stream_ = false;
  other.uses_unified_memory_ = false;
}

TensorRTExecutor& TensorRTExecutor::operator=(TensorRTExecutor&& other) noexcept {
  if (this != &other) {
    free_gpu_buffers();
    if (stream_ && owns_stream_) {
      cudaStreamDestroy(stream_);
    }
    runtime_ = std::move(other.runtime_);
    engine_ = std::move(other.engine_);
    context_ = std::move(other.context_);
    stream_ = other.stream_;
    owns_stream_ = other.owns_stream_;
    io_bindings_ = std::move(other.io_bindings_);
    gpu_buffers_ = std::move(other.gpu_buffers_);
    uses_unified_memory_ = other.uses_unified_memory_;

    other.stream_ = nullptr;
    other.owns_stream_ = false;
    other.uses_unified_memory_ = false;
  }
  return *this;
}

void TensorRTExecutor::set_cuda_stream(
    ::cudaStream_t stream,
    bool owns_stream) {
  if (stream_ && owns_stream_) {
    cudaStreamDestroy(stream_);
  }
  stream_ = stream;
  owns_stream_ = owns_stream;
}

Error TensorRTExecutor::initialize(
    const void* blob_data,
    size_t blob_size) {
  TensorRTBlobHeader header{};
  if (!parse_blob_header(blob_data, blob_size, header)) {
    ET_LOG(Error, "Failed to parse TensorRT blob header");
    return Error::InvalidArgument;
  }

  const void* engine_data = nullptr;
  size_t engine_size = 0;
  if (!get_engine_from_blob(blob_data, blob_size, header, engine_data, engine_size)) {
    ET_LOG(Error, "Failed to extract engine from blob");
    return Error::InvalidArgument;
  }

  const void* metadata_data = nullptr;
  size_t metadata_size = 0;
  if (header.metadata_size > 0) {
    if (!get_metadata_from_blob(blob_data, blob_size, header, metadata_data, metadata_size)) {
      ET_LOG(Info, "Failed to extract metadata from blob");
    } else {
      parse_io_bindings(metadata_data, metadata_size);
    }
  }

  // Initialize CUDA device before TensorRT
  int device_count = 0;
  cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "Failed to get CUDA device count: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidState;
  }
  if (device_count == 0) {
    ET_LOG(Error, "No CUDA devices available");
    return Error::InvalidState;
  }

  cuda_err = cudaSetDevice(0);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "Failed to set CUDA device: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidState;
  }

  ET_LOG(Info, "CUDA initialized with %d device(s)", device_count);

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(get_logger());
  if (!runtime) {
    ET_LOG(Error, "Failed to create TensorRT runtime");
    return Error::InvalidState;
  }
  runtime_.reset(runtime);

  nvinfer1::ICudaEngine* engine =
      runtime->deserializeCudaEngine(engine_data, engine_size);
  if (!engine) {
    ET_LOG(Error, "Failed to deserialize TensorRT engine");
    return Error::InvalidState;
  }
  engine_.reset(engine);

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context) {
    ET_LOG(Error, "Failed to create TensorRT execution context");
    return Error::InvalidState;
  }
  context_.reset(context);

  // Detect unified memory (Jetson and other integrated GPUs)
  cudaDeviceProp prop{};
  cuda_err = cudaGetDeviceProperties(&prop, 0);
  if (cuda_err == cudaSuccess) {
    uses_unified_memory_ = prop.integrated != 0;
    if (uses_unified_memory_) {
      ET_LOG(Info, "Detected integrated GPU with unified memory - skipping CPU-GPU copies");
    }
  }

  // Create persistent CUDA stream (unless an external stream was already set)
  if (stream_ == nullptr) {
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "Failed to create CUDA stream: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidState;
    }
    stream_ = stream;
    owns_stream_ = true;
  }

  // Pre-allocate GPU buffers
  // For unified memory (Jetson): use cudaMallocManaged
  // For discrete GPUs: use cudaMalloc
  auto alloc_err = allocate_gpu_buffers();
  if (alloc_err != Error::Ok) {
    return alloc_err;
  }

  ET_LOG(Info, "TensorRT executor initialized successfully");
  return Error::Ok;
}

Error TensorRTExecutor::allocate_gpu_buffers() {
  if (!engine_) {
    return Error::InvalidState;
  }

  const int32_t num_io_tensors = engine_->getNbIOTensors();

  gpu_buffers_.clear();
  gpu_buffers_.reserve(static_cast<size_t>(num_io_tensors));

  size_t input_idx = 0;
  size_t output_idx = 0;

  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    const auto mode = engine_->getTensorIOMode(name);
    const auto dims = engine_->getTensorShape(name);
    const auto dtype = engine_->getTensorDataType(name);

    size_t num_elems = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
      num_elems *= static_cast<size_t>(dims.d[d]);
    }
    const size_t buffer_size = num_elems * get_dtype_size(dtype);

    void* gpu_buffer = nullptr;
    cudaError_t cuda_err;
    if (uses_unified_memory_) {
      // Use managed memory for unified memory systems (Jetson)
      // This memory is accessible from both CPU and GPU
      cuda_err = cudaMallocManaged(&gpu_buffer, buffer_size);
    } else {
      // Use device memory for discrete GPUs
      cuda_err = cudaMalloc(&gpu_buffer, buffer_size);
    }
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "Failed to allocate GPU memory: %s", cudaGetErrorString(cuda_err));
      free_gpu_buffers();
      return Error::MemoryAllocationFailed;
    }

    GPUBuffer buf;
    buf.ptr = gpu_buffer;
    buf.size = buffer_size;
    buf.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
    buf.tensor_index = i;
    // Pre-compute the I/O index mapping to avoid runtime allocation
    if (buf.is_input) {
      buf.io_index = input_idx++;
    } else {
      buf.io_index = output_idx++;
    }
    gpu_buffers_.push_back(buf);
  }

  ET_LOG(Info, "Pre-allocated %zu %s buffers",
         gpu_buffers_.size(),
         uses_unified_memory_ ? "managed memory" : "GPU");
  return Error::Ok;
}

void TensorRTExecutor::free_gpu_buffers() {
  for (auto& buf : gpu_buffers_) {
    if (buf.ptr) {
      cudaFree(buf.ptr);
      buf.ptr = nullptr;
    }
  }
  gpu_buffers_.clear();
}

Error TensorRTExecutor::execute(
    void* const* input_buffers,
    size_t num_inputs,
    void* const* output_buffers,
    size_t num_outputs) {
  if (!is_initialized()) {
    ET_LOG(Error, "Executor not initialized");
    return Error::InvalidState;
  }

  // Validate buffer counts match expected (pre-computed during init)
  size_t expected_inputs = 0;
  size_t expected_outputs = 0;
  for (const auto& buf : gpu_buffers_) {
    if (buf.is_input) {
      ++expected_inputs;
    } else {
      ++expected_outputs;
    }
  }
  if (num_inputs < expected_inputs) {
    ET_LOG(Error, "Not enough input buffers: got %zu, expected %zu", num_inputs, expected_inputs);
    return Error::InvalidArgument;
  }
  if (num_outputs < expected_outputs) {
    ET_LOG(Error, "Not enough output buffers: got %zu, expected %zu", num_outputs, expected_outputs);
    return Error::InvalidArgument;
  }

  if (uses_unified_memory_) {
    // Unified memory path (Jetson): use cudaMallocManaged buffers
    // We must copy data to/from managed memory because ExecuTorch's planned
    // buffers are not CUDA-accessible. On unified memory systems, memcpy
    // is very fast as it's just a CPU-side copy within shared physical memory.
    for (const auto& buf : gpu_buffers_) {
      const char* name = engine_->getIOTensorName(buf.tensor_index);
      if (buf.is_input) {
        std::memcpy(buf.ptr, input_buffers[buf.io_index], buf.size);
      }
      context_->setTensorAddress(name, buf.ptr);
    }

    // Execute inference
    bool success = context_->enqueueV3(stream_);
    if (!success) {
      ET_LOG(Error, "TensorRT inference failed");
      return Error::InvalidState;
    }

    // Synchronize before reading outputs
    cudaError_t cuda_err = cudaStreamSynchronize(stream_);
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "CUDA synchronization failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidState;
    }

    // Copy outputs from managed memory
    for (const auto& buf : gpu_buffers_) {
      if (!buf.is_input) {
        std::memcpy(output_buffers[buf.io_index], buf.ptr, buf.size);
      }
    }
  } else {
    // Discrete GPU path: use pre-allocated GPU buffers with async copies.
    // cudaMemcpyDefault auto-detects pointer residency, so this works
    // correctly whether input/output buffers are on CPU (H2D/D2H) or
    // already on GPU (D2D), enabling unified backend pipelines where
    // inter-delegate data stays on the GPU.
    for (const auto& buf : gpu_buffers_) {
      const char* name = engine_->getIOTensorName(buf.tensor_index);
      if (buf.is_input) {
        cudaError_t cuda_err = cudaMemcpyAsync(
            buf.ptr,
            input_buffers[buf.io_index],
            buf.size,
            cudaMemcpyDefault,
            stream_);
        if (cuda_err != cudaSuccess) {
          ET_LOG(Error, "Failed to copy input to GPU: %s", cudaGetErrorString(cuda_err));
          return Error::InvalidState;
        }
      }
      context_->setTensorAddress(name, buf.ptr);
    }

    // Execute inference
    bool success = context_->enqueueV3(stream_);
    if (!success) {
      ET_LOG(Error, "TensorRT inference failed");
      return Error::InvalidState;
    }

    // Copy outputs from GPU
    for (const auto& buf : gpu_buffers_) {
      if (!buf.is_input) {
        cudaError_t cuda_err = cudaMemcpyAsync(
            output_buffers[buf.io_index],
            buf.ptr,
            buf.size,
            cudaMemcpyDefault,
            stream_);
        if (cuda_err != cudaSuccess) {
          ET_LOG(Error, "Failed to copy output from GPU: %s", cudaGetErrorString(cuda_err));
          return Error::InvalidState;
        }
      }
    }

    // Synchronize
    cudaError_t cuda_err = cudaStreamSynchronize(stream_);
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "CUDA synchronization failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidState;
    }
  }

  return Error::Ok;
}

bool TensorRTExecutor::parse_io_bindings(const void* json_data, size_t json_size) {
  (void)json_data;
  (void)json_size;
  // TODO: Implement JSON parsing for I/O bindings
  return true;
}

size_t TensorRTExecutor::get_num_inputs() const {
  if (!engine_) {
    return 0;
  }
  const int32_t num_io_tensors = engine_->getNbIOTensors();
  size_t count = 0;
  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      ++count;
    }
  }
  return count;
}

size_t TensorRTExecutor::get_num_outputs() const {
  if (!engine_) {
    return 0;
  }
  const int32_t num_io_tensors = engine_->getNbIOTensors();
  size_t count = 0;
  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      ++count;
    }
  }
  return count;
}

} // namespace tensorrt
} // namespace backends
} // namespace executorch
