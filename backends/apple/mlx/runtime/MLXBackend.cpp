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
#include <iostream>
#include <limits>
#include <memory>

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
    shape.push_back(static_cast<int>(t.size(i)));
  }

  void* data_ptr = const_cast<void*>(t.const_data_ptr());
  auto deleter = [](void*) {};
  return array(data_ptr, shape, dtype, deleter);
}

void mlx_to_tensor(const array& arr, ETTensor& out) {
  array contiguous_arr = ::mlx::core::contiguous(arr);

  Dtype expected_dtype = resolve_dtype(
      static_cast<executorch::aten::ScalarType>(out.scalar_type()));

  if (contiguous_arr.dtype() != expected_dtype) {
    contiguous_arr = ::mlx::core::astype(contiguous_arr, expected_dtype);
  }

  eval(contiguous_arr);

  // Resize output tensor if shape doesn't match (dynamic shapes)
  const auto& mlx_shape = contiguous_arr.shape();
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
    std::vector<executorch::aten::SizesType> new_sizes(
        mlx_shape.begin(), mlx_shape.end());
    auto err = resize_tensor(
        out,
        ArrayRef<executorch::aten::SizesType>(
            new_sizes.data(), new_sizes.size()));
    if (err != Error::Ok) {
      throw std::runtime_error("mlx_to_tensor: failed to resize output tensor");
    }
  }

  size_t mlx_nbytes = contiguous_arr.nbytes();
  size_t out_nbytes = out.nbytes();
  if (mlx_nbytes != out_nbytes) {
    throw std::runtime_error(
        "mlx_to_tensor: size mismatch - MLX has " + std::to_string(mlx_nbytes) +
        " bytes, output has " + std::to_string(out_nbytes) + " bytes");
  }

  std::memcpy(out.mutable_data_ptr(), contiguous_arr.data<void>(), out_nbytes);
}

} // namespace

struct MLXHandle {
  MLXProgram program;
  ConstantData constants;
  MutableBufferData mutable_buffers;
  ExecutionState state; // Reusable execution state
  Interpreter interpreter;

  // Keep the constant buffers alive for zero-copy constants
  // Each FreeableBuffer must outlive the MLX arrays that reference it
  std::vector<FreeableBuffer> constant_buffers;

  MLXHandle() = default;
  ~MLXHandle() = default;

  MLXHandle(const MLXHandle&) = delete;
  MLXHandle& operator=(const MLXHandle&) = delete;
};

class MLXBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~MLXBackend() override = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    auto* handle =
        context.get_runtime_allocator()->allocateInstance<MLXHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    try {
      new (handle) MLXHandle();

      handle->program = loader::load_program(
          static_cast<const uint8_t*>(processed->data()), processed->size());

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

    } catch (const std::exception& e) {
      ET_LOG(Error, "Failed to load MLX program: %s", e.what());
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
      auto* h = static_cast<MLXHandle*>(handle);
      const auto& program = h->program;

      // Reset per-execution state (clears inputs/outputs/temps, keeps mutable
      // buffers)
      h->state.reset();

      // Walk args in lockstep with input_map, then output_map.
      // Since mutable buffers are not in the maps, args correspond 1:1 with map
      // entries.
      size_t arg_idx = 0;
      const size_t expected_args =
          program.input_map.size() + program.output_map.size();
      if (args.size() != expected_args) {
        ET_LOG(Error, "Expected %zu args, got %zu", expected_args, args.size());
        return Error::InvalidArgument;
      }

      // --- Bind inputs ---
      for (const auto& slot : program.input_map) {
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
          h->state.set_value(Vid<int32_t>{slot.idx}, static_cast<int32_t>(val));
        } else {
          throw std::runtime_error(
              "Unhandled input slot type: " +
              std::to_string(static_cast<int>(slot.slot_type)));
        }
      }

      // --- Run the MLX program ---
      h->interpreter.run(program, h->state);

      // --- Collect tensor outputs for batch eval ---
      std::vector<array> tensor_arrays;
      tensor_arrays.reserve(program.num_output_tensors);
      for (const auto& slot : program.output_map) {
        if (slot.slot_type == SlotType::TensorSlot) {
          tensor_arrays.push_back(h->state.const_tensor_ref(Tid{slot.idx}));
        }
      }
      eval(tensor_arrays);

      // --- Write outputs back to args ---
      size_t tensor_out_idx = 0;
      for (const auto& slot : program.output_map) {
        if (slot.slot_type == SlotType::TensorSlot) {
          ETTensor& out_tensor = args[arg_idx++]->toTensor();
          mlx_to_tensor(tensor_arrays[tensor_out_idx++], out_tensor);
        } else if (slot.slot_type == SlotType::IntValueSlot) {
          Vid<int32_t> vid{slot.idx};
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

      return Error::Ok;
    } catch (const std::exception& e) {
      ET_LOG(Error, "MLX execute failed: %s", e.what());
      return Error::Internal;
    }
  }

  void destroy(DelegateHandle* handle) const override {
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
