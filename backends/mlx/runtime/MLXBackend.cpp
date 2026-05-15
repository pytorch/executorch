//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#include "MLXExecutor.h"
#include "MLXInterpreter.h"
#include "MLXLoader.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>

#include <mlx/mlx.h>

#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace executorch {
namespace backends {
namespace mlx {

// Note: We use fully qualified executorch::aten::Tensor because MLXExecutor.h
// defines Tensor as mlx::core::array in the executorch::backends::mlx
// namespace.
using ETTensor = ::executorch::aten::Tensor;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

using ::mlx::core::array;
using ::mlx::core::Dtype;
using ::mlx::core::eval;

namespace {

array tensor_to_mlx(
    const ETTensor& t,
    const std::optional<TensorMeta>& expected_meta = std::nullopt) {
  if (!executorch::runtime::tensor_is_contiguous(t)) {
    throw std::runtime_error("tensor_to_mlx: input tensor is not contiguous");
  }

  Dtype dtype =
      resolve_dtype(static_cast<executorch::aten::ScalarType>(t.scalar_type()));

  if (expected_meta.has_value()) {
    Dtype expected_dtype = resolve_dtype(expected_meta->scalar_type);
    if (dtype != expected_dtype) {
      throw std::runtime_error(
          std::string("tensor_to_mlx: dtype mismatch - input tensor has ") +
          ExecutionState::dtype_str(dtype) + " but model expects " +
          ExecutionState::dtype_str(expected_dtype));
    }
  }

  ::mlx::core::Shape shape;
  for (int i = 0; i < t.dim(); ++i) {
    auto dim_size = t.size(i);
    if (dim_size > std::numeric_limits<int>::max() ||
        dim_size < std::numeric_limits<int>::min()) {
      throw std::runtime_error(
          "tensor_to_mlx: dimension " + std::to_string(i) + " size " +
          std::to_string(dim_size) + " exceeds int range");
    }
    shape.push_back(static_cast<int>(dim_size));
  }

  // SAFETY: MLX reads this data during async_eval() Metal command encoding,
  // which completes before the lock is released. The ET tensor must remain
  // valid until async_eval returns.
  const void* cptr = t.const_data_ptr();
  if (!cptr) {
    throw std::runtime_error("tensor_to_mlx: tensor has null data pointer");
  }
  void* data_ptr = const_cast<void*>(cptr);
  auto deleter = [](void*) {};
  return array(data_ptr, shape, dtype, deleter);
}

// Build the contiguous + dtype conversion pipeline for an output array.
// Returns a lazy array (not yet evaluated) ready for async_eval.
array prepare_output(
    const array& arr,
    Dtype expected_dtype,
    const ::mlx::core::Stream& stream) {
  array result =
      ::mlx::core::contiguous(arr, /*allow_col_major=*/false, stream);
  if (result.dtype() != expected_dtype) {
    result = ::mlx::core::astype(result, expected_dtype, stream);
  }
  return result;
}

// Wait for a prepared output array and copy its data to an ET tensor.
// The array must have been submitted via async_eval before calling this.
void write_output(array& arr, ETTensor& out) {
  arr.wait();

  // Resize output tensor if shape doesn't match (dynamic shapes)
  const auto& mlx_shape = arr.shape();
  auto out_sizes = out.sizes();

  bool shape_matches = (mlx_shape.size() == static_cast<size_t>(out.dim()));
  if (shape_matches) {
    for (size_t i = 0; i < mlx_shape.size(); ++i) {
      if (static_cast<int64_t>(mlx_shape[i]) !=
          static_cast<int64_t>(out_sizes[i])) {
        shape_matches = false;
        break;
      }
    }
  }

  if (!shape_matches) {
    std::vector<executorch::aten::SizesType> new_sizes;
    new_sizes.reserve(mlx_shape.size());
    for (auto d : mlx_shape) {
      new_sizes.push_back(static_cast<executorch::aten::SizesType>(d));
    }
    auto err = resize_tensor(
        out,
        ArrayRef<executorch::aten::SizesType>(
            new_sizes.data(), new_sizes.size()));
    if (err != Error::Ok) {
      throw std::runtime_error("write_output: failed to resize output tensor");
    }
  }

  size_t mlx_nbytes = arr.nbytes();
  size_t out_nbytes = out.nbytes();
  if (mlx_nbytes != out_nbytes) {
    throw std::runtime_error(
        "write_output: size mismatch - MLX has " + std::to_string(mlx_nbytes) +
        " bytes, output has " + std::to_string(out_nbytes) + " bytes");
  }

  const void* src = arr.data<void>();
  if (!src) {
    throw std::runtime_error(
        "write_output: arr.data<void>() is null after wait()");
  }
  std::memcpy(out.mutable_data_ptr(), src, out_nbytes);
}

} // namespace

struct MLXHandle {
  MLXProgram program;
  ConstantData constants;
  MutableBufferData mutable_buffers;
  ExecutionState state; // Reusable execution state
  Interpreter interpreter;
  ::mlx::core::Stream stream; // Dedicated GPU stream for this handle

  // Keep the constant buffers alive for zero-copy constants
  // Each FreeableBuffer must outlive the MLX arrays that reference it
  std::vector<FreeableBuffer> constant_buffers;

  MLXHandle() : stream(::mlx::core::new_stream(::mlx::core::Device::gpu)) {}
  ~MLXHandle() = default;

  MLXHandle(const MLXHandle&) = delete;
  MLXHandle& operator=(const MLXHandle&) = delete;
};

// MLX is not thread-safe: its computation graph is global shared state.
// A global mutex serializes graph construction and command submission
// across all handles. GPU execution and output copies can proceed
// without the lock (see execute() for the async pipeline design).
static std::mutex& mlx_global_mutex() {
  static std::mutex m;
  return m;
}

class MLXBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~MLXBackend() override = default;

  bool is_available() const override {
    return ::mlx::core::metal::is_available();
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    std::lock_guard<std::mutex> lock(mlx_global_mutex());
    auto* handle =
        context.get_runtime_allocator()->allocateInstance<MLXHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    try {
      new (handle) MLXHandle();

      if (!processed || !processed->data() || processed->size() == 0) {
        throw std::runtime_error("init: null or empty delegate payload");
      }

      handle->program = loader::load_program(
          static_cast<const uint8_t*>(processed->data()), processed->size());

      // Validate schema version
      int schema_version = 1;
      if (!handle->program.version.empty()) {
        try {
          schema_version = std::stoi(handle->program.version);
        } catch (...) {
          throw std::runtime_error(
              "Invalid MLX schema version '" + handle->program.version +
              "' (expected integer)");
        }
      }
      constexpr int kMaxSupportedVersion = 1;
      if (schema_version > kMaxSupportedVersion) {
        throw std::runtime_error(
            "This .pte requires ExecuTorch MLX runtime version " +
            std::to_string(schema_version) +
            " but this runtime only supports up to version " +
            std::to_string(kMaxSupportedVersion) +
            ". Upgrade ExecuTorch to a newer version.");
      }

      // Load constants from named_data_map
      // Constants are stored by name in the .pte file and provided by ET at
      // runtime
      const runtime::NamedDataMap* named_data_map =
          context.get_named_data_map();
      load_constants(
          handle->program,
          named_data_map,
          handle->constants,
          handle->constant_buffers);

      // Delegate payload no longer needed after constants are loaded
      processed->Free();
      processed = nullptr;

      // Load mutable buffers (e.g., KV cache)
      load_mutable_buffers(handle->program, handle->mutable_buffers);

      // Bind execution state (reused across execute() calls)
      handle->state.bind(
          handle->program, handle->constants, handle->mutable_buffers);

      // Run init chain if present.
      // SAFETY: The >= 0 check ensures init_chain_idx is non-negative, so the
      // static_cast<uint32_t> cannot produce UINT32_MAX from a -1 sentinel.
      if (handle->program.init_chain_idx >= 0) {
        handle->state.is_init_chain = true;
        handle->interpreter.run_chain(
            handle->program,
            static_cast<uint32_t>(handle->program.init_chain_idx),
            handle->state,
            handle->stream);
        handle->state.is_init_chain = false;

        // Evaluate any constants written by the init chain so the first
        // execute() doesn't pay the cost of materializing them.
        eval(handle->constants.tensors);
      }

    } catch (const std::exception& e) {
      ET_LOG(Error, "Failed to load MLX program: %s", e.what());
      handle->~MLXHandle();
      if (processed != nullptr) {
        processed->Free();
      }
      return Error::InvalidProgram;
    } catch (...) {
      ET_LOG(Error, "Failed to load MLX program: unknown non-std exception");
      handle->~MLXHandle();
      if (processed != nullptr) {
        processed->Free();
      }
      return Error::InvalidProgram;
    }

    return handle;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    try {
      std::vector<array> prepared_outputs;
      struct OutputInfo {
        size_t arg_idx;
        size_t prepared_idx;
      };

      std::vector<OutputInfo> tensor_output_info;
      size_t arg_idx = 0;

      auto* h = static_cast<MLXHandle*>(handle);
      const auto& program = h->program;

      // Graph construction + async GPU dispatch (locked)
      {
        std::lock_guard<std::mutex> lock(mlx_global_mutex());

        h->state.reset();

        const size_t n_inputs = program.input_map.size();
        const size_t n_outputs = program.output_map.size();
        if (n_inputs > SIZE_MAX - n_outputs) {
          throw std::runtime_error("execute: input + output count overflow");
        }
        const size_t expected_args = n_inputs + n_outputs;
        if (args.size() != expected_args) {
          ET_LOG(
              Error, "Expected %zu args, got %zu", expected_args, args.size());
          return Error::InvalidArgument;
        }

        // Bind inputs
        for (const auto& slot : program.input_map) {
          if (arg_idx >= args.size()) {
            throw std::runtime_error(
                "execute: arg_idx " + std::to_string(arg_idx) +
                " out of bounds (args.size()=" + std::to_string(args.size()) +
                ")");
          }
          if (slot.slot_type == SlotType::TensorSlot) {
            const ETTensor& tensor = args[arg_idx++]->toTensor();
            Tid tid{slot.idx};
            std::optional<TensorMeta> expected_meta = std::nullopt;
            if (tid.idx < program.tensor_meta.size()) {
              expected_meta = program.tensor_meta[tid.idx];
            }
            h->state.set_tensor(tid, tensor_to_mlx(tensor, expected_meta));
          } else if (slot.slot_type == SlotType::IntValueSlot) {
            int64_t val = args[arg_idx]->toInt();
            arg_idx++;
            if (val > std::numeric_limits<int32_t>::max() ||
                val < std::numeric_limits<int32_t>::min()) {
              ET_LOG(
                  Error,
                  "Int input value %lld exceeds int32 range",
                  static_cast<long long>(val));
              return Error::InvalidArgument;
            }
            h->state.set_value(Vid{slot.idx}, static_cast<int32_t>(val));
          } else {
            throw std::runtime_error(
                "Unhandled input slot type: " +
                std::to_string(static_cast<int>(slot.slot_type)));
          }
        }

        // Run the MLX program (builds lazy computation graph)
        h->interpreter.run(program, h->state, h->stream);

        // Prepare output pipeline and collect int outputs
        // Build contiguous + dtype conversion lazily for each tensor output,
        // and extract int outputs (which don't need GPU) while still locked.
        prepared_outputs.reserve(program.num_output_tensors);

        for (const auto& slot : program.output_map) {
          if (slot.slot_type == SlotType::TensorSlot) {
            ETTensor& out_tensor = args[arg_idx]->toTensor();
            Dtype expected_dtype =
                resolve_dtype(static_cast<executorch::aten::ScalarType>(
                    out_tensor.scalar_type()));
            array out_arr = prepare_output(
                h->state.const_tensor_ref(Tid{slot.idx}),
                expected_dtype,
                h->stream);
            tensor_output_info.push_back({arg_idx, prepared_outputs.size()});
            prepared_outputs.push_back(std::move(out_arr));
            arg_idx++;
          } else if (slot.slot_type == SlotType::IntValueSlot) {
            Vid vid{slot.idx};
            int64_t int_val =
                static_cast<int64_t>(h->state.const_value_ref<int32_t>(vid));
            *args[arg_idx] = EValue(int_val);
            arg_idx++;
          } else {
            throw std::runtime_error(
                "Unhandled output slot type: " +
                std::to_string(static_cast<int>(slot.slot_type)));
          }
        }

        // Submit all output work to GPU asynchronously
        // async_eval encodes Metal commands and returns immediately.
        // The GPU will signal events on completion.
        if (!prepared_outputs.empty()) {
          ::mlx::core::async_eval(prepared_outputs);
        }

      } // Lock released — GPU is still executing

      for (auto& info : tensor_output_info) {
        ETTensor& out_tensor = args[info.arg_idx]->toTensor();

        // write_output waits on arr to be ready
        write_output(prepared_outputs[info.prepared_idx], out_tensor);
      }

      h->state.reset(); // Release temp GPU buffers back to MLX cache

      return Error::Ok;
    } catch (const std::exception& e) {
      ET_LOG(Error, "MLX execute failed: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "MLX execute failed: unknown non-std exception");
      return Error::Internal;
    }
  }

  void destroy(DelegateHandle* handle) const override {
    std::lock_guard<std::mutex> lock(mlx_global_mutex());
    if (handle != nullptr) {
      auto* mlx_handle = static_cast<MLXHandle*>(handle);
      mlx_handle->~MLXHandle();
    }
  }
};

namespace {
auto cls = MLXBackend();
Backend backend{"MLXBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace mlx
} // namespace backends
} // namespace executorch
