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

#include <mlx/mlx.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_set>

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

Dtype et_dtype_to_mlx(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:
      return ::mlx::core::float32;
    case executorch::aten::ScalarType::Half:
      return ::mlx::core::float16;
    case executorch::aten::ScalarType::BFloat16:
      return ::mlx::core::bfloat16;
    case executorch::aten::ScalarType::Int:
      return ::mlx::core::int32;
    case executorch::aten::ScalarType::Long:
      return ::mlx::core::int64;
    case executorch::aten::ScalarType::Short:
      return ::mlx::core::int16;
    case executorch::aten::ScalarType::Byte:
      return ::mlx::core::uint8;
    case executorch::aten::ScalarType::Char:
      return ::mlx::core::int8;
    case executorch::aten::ScalarType::Bool:
      return ::mlx::core::bool_;
    default:
      ET_LOG(Error, "Unsupported dtype %d", static_cast<int>(dtype));
      return ::mlx::core::float32;
  }
}

std::vector<int> shape_to_vector(const ETTensor& t) {
  std::vector<int> shape;
  shape.reserve(t.dim());
  for (int i = 0; i < t.dim(); ++i) {
    shape.push_back(static_cast<int>(t.size(i)));
  }
  return shape;
}

array tensor_to_mlx(const ETTensor& t) {
  auto dtype = et_dtype_to_mlx(t.scalar_type());

  // Convert shape to MLX Shape type
  ::mlx::core::Shape shape;
  for (int i = 0; i < t.dim(); ++i) {
    shape.push_back(static_cast<int>(t.size(i)));
  }

  // Create MLX array from raw CPU data
  // MLX will copy the data to Metal-aligned memory
  const void* data_ptr = t.const_data_ptr();
  size_t nbytes = t.nbytes();

  // Create an MLX array by copying data from the CPU pointer
  // We need to use the appropriate typed constructor based on dtype
  switch (dtype) {
    case ::mlx::core::float32:
      return array(static_cast<const float*>(data_ptr), shape, dtype);
    case ::mlx::core::float16:
      return array(
          static_cast<const ::mlx::core::float16_t*>(data_ptr), shape, dtype);
    case ::mlx::core::bfloat16:
      return array(
          static_cast<const ::mlx::core::bfloat16_t*>(data_ptr), shape, dtype);
    case ::mlx::core::int32:
      return array(static_cast<const int32_t*>(data_ptr), shape, dtype);
    case ::mlx::core::int64:
      return array(static_cast<const int64_t*>(data_ptr), shape, dtype);
    case ::mlx::core::int16:
      return array(static_cast<const int16_t*>(data_ptr), shape, dtype);
    case ::mlx::core::int8:
      return array(static_cast<const int8_t*>(data_ptr), shape, dtype);
    case ::mlx::core::uint8:
      return array(static_cast<const uint8_t*>(data_ptr), shape, dtype);
    case ::mlx::core::bool_:
      return array(static_cast<const bool*>(data_ptr), shape, dtype);
    default:
      // Fallback: treat as float
      return array(static_cast<const float*>(data_ptr), shape, dtype);
  }
}

void mlx_to_tensor(const array& arr, ETTensor& out) {
  array contiguous_arr = ::mlx::core::contiguous(arr);
  eval(contiguous_arr);

  // Update output tensor shape to match actual MLX output shape (for dynamic
  // shapes)
  const auto& mlx_shape = contiguous_arr.shape();
  auto out_sizes = out.sizes();

  // Check if shapes match; if not, we need to resize the output
  bool shape_matches = (mlx_shape.size() == static_cast<size_t>(out.dim()));
  if (shape_matches) {
    for (size_t i = 0; i < mlx_shape.size(); ++i) {
      if (mlx_shape[i] != out_sizes[i]) {
        shape_matches = false;
        break;
      }
    }
  }

  if (!shape_matches) {
    // Create new sizes array for resize
    std::vector<executorch::aten::SizesType> new_sizes(
        mlx_shape.begin(), mlx_shape.end());
    auto err = resize_tensor(
        out,
        ArrayRef<executorch::aten::SizesType>(
            new_sizes.data(), new_sizes.size()));
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to resize output tensor for dynamic shape");
      // Fall through - will copy what we can
    }
  }

  void* out_ptr = out.mutable_data_ptr();
  size_t nbytes = contiguous_arr.nbytes();
  std::memcpy(out_ptr, contiguous_arr.data<void>(), nbytes);
}

} // namespace

struct MLXHandle {
  MLXProgram program;
  ConstantData constants;
  MutableBufferData mutable_buffers;
  Interpreter interpreter;

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

    new (handle) MLXHandle();

    try {
      handle->program = loader::load_program(
          static_cast<const uint8_t*>(processed->data()), processed->size());

      load_constants(handle->program, handle->constants);
      load_mutable_buffers(handle->program, handle->mutable_buffers);
    } catch (const std::exception& e) {
      ET_LOG(Error, "Failed to load MLX program: %s", e.what());
      handle->~MLXHandle();
      return Error::InvalidProgram;
    }

    processed->Free();

    return handle;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    auto* mlx_handle = static_cast<MLXHandle*>(handle);
    const auto& program = mlx_handle->program;

    ExecutionState state;
    state.bind(program, mlx_handle->constants);

    size_t num_inputs = program.input_map.size();
    size_t num_outputs = program.output_map.size();
    size_t num_mutable_buffers = program.mutable_buffer_map.size();

    // Build a set of mutable buffer tensor IDs for quick lookup
    std::unordered_set<uint32_t> mutable_buffer_tids;
    for (const auto& slot : program.mutable_buffer_map) {
      if (slot.slot_type == SlotType::TensorSlot) {
        mutable_buffer_tids.insert(slot.idx);
      }
    }

    // Count tensor and int outputs (excluding mutable buffer mutations)
    size_t num_tensor_outputs = 0;
    size_t num_int_outputs = 0;
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& slot = program.output_map[i];
      if (slot.slot_type == SlotType::TensorSlot) {
        // Check if this is a mutable buffer (BUFFER_MUTATION output)
        // Mutable buffer outputs don't need ExecuTorch output tensors - they're
        // updated in-place
        if (mutable_buffer_tids.find(slot.idx) == mutable_buffer_tids.end()) {
          num_tensor_outputs++;
        }
      } else if (slot.slot_type == SlotType::IntValueSlot) {
        num_int_outputs++;
      }
    }

    // Count regular inputs by type (excluding mutable buffers which are
    // delegate-owned)
    size_t num_regular_tensor_inputs = 0;
    size_t num_regular_int_inputs = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& slot = program.input_map[i];
      if (slot.slot_type == SlotType::TensorSlot) {
        // Skip if this is a mutable buffer
        if (mutable_buffer_tids.find(slot.idx) == mutable_buffer_tids.end()) {
          num_regular_tensor_inputs++;
        }
      } else if (slot.slot_type == SlotType::IntValueSlot) {
        num_regular_int_inputs++;
      }
    }

    // Collect tensor and int args from ExecuTorch
    std::vector<const ETTensor*> input_tensors;
    std::vector<int64_t> input_ints;
    std::vector<ETTensor*> output_tensors;
    std::vector<EValue*> output_ints;

    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] == nullptr) {
        continue;
      }

      if (args[i]->isTensor()) {
        if (input_tensors.size() < num_regular_tensor_inputs) {
          input_tensors.push_back(&args[i]->toTensor());
        } else if (output_tensors.size() < num_tensor_outputs) {
          output_tensors.push_back(&args[i]->toTensor());
        }
      } else if (args[i]->isInt()) {
        // Int args: first num_regular_int_inputs are inputs, rest are outputs
        if (input_ints.size() < num_regular_int_inputs) {
          input_ints.push_back(args[i]->toInt());
        } else if (output_ints.size() < num_int_outputs) {
          output_ints.push_back(args[i]);
        }
      } else if (args[i]->isTensorList()) {
        auto tensor_list = args[i]->toTensorList();
        for (auto& tensor : tensor_list) {
          if (input_tensors.size() < num_regular_tensor_inputs) {
            input_tensors.push_back(&tensor);
          } else if (output_tensors.size() < num_tensor_outputs) {
            output_tensors.push_back(const_cast<ETTensor*>(&tensor));
          }
        }
      }
    }

    if (input_tensors.size() != num_regular_tensor_inputs) {
      ET_LOG(
          Error,
          "Expected %zu regular tensor inputs, got %zu",
          num_regular_tensor_inputs,
          input_tensors.size());
      return Error::InvalidArgument;
    }
    if (input_ints.size() != num_regular_int_inputs) {
      ET_LOG(
          Error,
          "Expected %zu int inputs, got %zu",
          num_regular_int_inputs,
          input_ints.size());
      return Error::InvalidArgument;
    }
    if (output_tensors.size() != num_tensor_outputs) {
      ET_LOG(
          Error,
          "Expected %zu tensor outputs, got %zu",
          num_tensor_outputs,
          output_tensors.size());
      return Error::InvalidArgument;
    }

    // Bind inputs to state
    // First, bind mutable buffers from delegate-owned storage
    for (const auto& slot : program.mutable_buffer_map) {
      if (slot.slot_type == SlotType::TensorSlot) {
        Tid tid{slot.idx};
        // Get the delegate-owned MLX array (persists across executions)
        array& arr = mlx_handle->mutable_buffers.get(tid);
        state.set_tensor(tid, arr); // Copy to state
      }
    }

    // Then, bind regular inputs from ExecuTorch
    size_t regular_tensor_idx = 0;
    size_t regular_int_idx = 0;

    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& slot = program.input_map[i];
      if (slot.slot_type == SlotType::TensorSlot) {
        Tid tid{slot.idx};
        // Skip if this is a mutable buffer (already bound above)
        if (mutable_buffer_tids.find(slot.idx) != mutable_buffer_tids.end()) {
          continue;
        }
        // Bind regular input from ExecuTorch
        array arr = tensor_to_mlx(*input_tensors[regular_tensor_idx]);
        state.set_tensor(tid, std::move(arr));
        regular_tensor_idx++;
      } else if (slot.slot_type == SlotType::IntValueSlot) {
        // Bind int value input from ExecuTorch
        Vid<int32_t> vid{slot.idx};
        if (regular_int_idx < input_ints.size()) {
          state.set_value(
              vid, static_cast<int32_t>(input_ints[regular_int_idx]));
          regular_int_idx++;
        } else {
          ET_LOG(Error, "Missing int input for slot %zu", i);
          return Error::InvalidArgument;
        }
      } else {
        ET_LOG(
            Error,
            "Input slot %zu has unsupported type %d",
            i,
            static_cast<int>(slot.slot_type));
        return Error::InvalidProgram;
      }
    }

    // Run the MLX program
    try {
      mlx_handle->interpreter.run(program, state);
    } catch (const std::exception& e) {
      ET_LOG(Error, "MLX execution failed: %s", e.what());
      return Error::Internal;
    }

    // After execution, update delegate-owned mutable buffers with the results
    // This is needed because slice_update returns a new array
    for (const auto& slot : program.mutable_buffer_map) {
      if (slot.slot_type == SlotType::TensorSlot) {
        Tid tid{slot.idx};
        // Get the updated tensor from execution state and store it back
        array updated = state.const_tensor_ref(tid);
        mlx_handle->mutable_buffers.set(tid, std::move(updated));
      }
    }

    // Collect regular tensor outputs (not mutable buffers) for eval
    std::vector<array> tensor_arrays;
    tensor_arrays.reserve(num_tensor_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& slot = program.output_map[i];
      if (slot.slot_type == SlotType::TensorSlot) {
        Tid tid{slot.idx};
        // Skip mutable buffer outputs - they don't have ExecuTorch output
        // tensors
        if (mutable_buffer_tids.find(slot.idx) != mutable_buffer_tids.end()) {
          continue;
        }
        tensor_arrays.push_back(state.const_tensor_ref(tid));
      }
    }

    // Evaluate all tensor outputs
    eval(tensor_arrays);

    // Write tensor outputs to ExecuTorch
    for (size_t i = 0; i < num_tensor_outputs; ++i) {
      mlx_to_tensor(tensor_arrays[i], *output_tensors[i]);
    }

    // Write int outputs if we have them
    size_t int_idx = 0;
    for (size_t i = 0; i < num_outputs && int_idx < output_ints.size(); ++i) {
      const auto& slot = program.output_map[i];
      if (slot.slot_type == SlotType::IntValueSlot) {
        Vid<int32_t> vid{slot.idx};
        int64_t int_val =
            static_cast<int64_t>(state.const_value_ref<int32_t>(vid));
        *output_ints[int_idx] = EValue(int_val);
        int_idx++;
      }
    }

    return Error::Ok;
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
