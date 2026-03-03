//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#pragma once

#include "MLXExecutor.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace ops {

using namespace ::mlx::core;

/**
 * Normalize axis to be in range [0, rank) and validate.
 * @param axis The axis value (can be negative)
 * @param rank The tensor rank
 * @param op_name Name of the operation for error messages
 * @return Normalized axis in range [0, rank)
 * @throws std::out_of_range if axis is out of range
 */
inline int normalize_axis(int axis, int rank, const char* op_name) {
  if (axis < -rank || axis >= rank) {
    throw std::out_of_range(std::string(op_name) + ": axis out of range");
  }
  if (axis < 0)
    axis += rank;
  return axis;
}

/**
 * Infers dimensions with -1 in a reshape-like operation.
 *
 * PyTorch allows -1 in shapes to mean "infer this dimension from total size".
 * MLX requires concrete positive integers, so we must resolve -1 values.
 *
 * @param shape The shape to resolve (may contain -1)
 * @param input_size Total number of elements in the input tensor
 * @return Resolved shape with all positive integers
 * @throws std::runtime_error if shape has multiple -1 or incompatible sizes
 */
inline std::vector<int> infer_shape_with_minus_one(
    const std::vector<int>& shape,
    size_t input_size) {
  std::vector<int> resolved_shape = shape;
  int neg_one_idx = -1;
  int64_t known_size = 1; // Use int64_t to avoid overflow

  // Find -1 dimension and compute product of known dimensions
  for (size_t i = 0; i < resolved_shape.size(); i++) {
    if (resolved_shape[i] == -1) {
      if (neg_one_idx != -1) {
        throw std::runtime_error("infer_shape: only one dimension can be -1");
      }
      neg_one_idx = static_cast<int>(i);
    } else {
      known_size *= static_cast<int64_t>(resolved_shape[i]);
    }
  }

  // Infer the -1 dimension if present
  if (neg_one_idx != -1) {
    if (known_size == 0) {
      throw std::runtime_error(
          "infer_shape: cannot infer -1 dimension when known product is 0");
    }
    int64_t input_size_i64 = static_cast<int64_t>(input_size);
    if (input_size_i64 % known_size != 0) {
      throw std::runtime_error(
          "infer_shape: cannot infer dimension - size mismatch");
    }
    int64_t inferred_dim = input_size_i64 / known_size;

    // Check that inferred dimension fits in int
    if (inferred_dim > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "infer_shape: inferred dimension exceeds int max");
    }

    resolved_shape[static_cast<size_t>(neg_one_idx)] =
        static_cast<int>(inferred_dim);
  }

  return resolved_shape;
}

inline void exec_noop(const NoopNode&, ExecutionState&, StreamOrDevice) {}

inline void
exec_id_copy(const IdCopyNode& n, ExecutionState& st, StreamOrDevice) {
  st.set_tensor(n.out, st.const_tensor_ref(n.x));
}

inline void
exec_addmm(const AddmmNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& mat1 = st.const_tensor_ref(n.mat1);
  const auto& mat2 = st.const_tensor_ref(n.mat2);

  array Y = n.bias ? addmm(
                         st.const_tensor_ref(*n.bias),
                         mat1,
                         mat2,
                         /*alpha=*/n.alpha,
                         /*beta=*/n.beta,
                         s)
                   : matmul(mat1, mat2, s);

  st.set_tensor(n.out, std::move(Y));
}

} // namespace ops

class Interpreter {
 public:
  void run(
      const MLXProgram& prog,
      ExecutionState& st,
      StreamOrDevice stream = {}) const {
    run_chain(prog, prog.main_chain_idx, st, stream);
  }

  void run_chain(
      const MLXProgram& prog,
      uint32_t chain_idx,
      ExecutionState& st,
      StreamOrDevice stream = {}) const {
    if (chain_idx >= prog.instruction_chains.size()) {
      throw std::runtime_error(
          "run_chain: chain_idx " + std::to_string(chain_idx) +
          " out of range (num_chains=" +
          std::to_string(prog.instruction_chains.size()) + ")");
    }
    const auto& chain = prog.instruction_chains[chain_idx];
    size_t idx = 0;
    for (const auto& instr : chain) {
      st.begin_op(idx, op_name(instr.op));
      dispatch(instr, st, stream);
      st.end_op();
      ++idx;
    }
  }

 private:
  void dispatch(const Instruction& instr, ExecutionState& st, StreamOrDevice s)
      const {
    switch (instr.op) {
      case OpCode::NOOP:
        ops::exec_noop(std::get<NoopNode>(instr.node), st, s);
        break;
      case OpCode::ID_COPY:
        ops::exec_id_copy(std::get<IdCopyNode>(instr.node), st, s);
        break;
      case OpCode::ADDMM:
        ops::exec_addmm(std::get<AddmmNode>(instr.node), st, s);
        break;
      default:
        throw std::runtime_error(
            "Unknown opcode: " + std::to_string(static_cast<int>(instr.op)));
    }
  }
};

} // namespace mlx
} // namespace backends
} // namespace executorch
