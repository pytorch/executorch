/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ggml.h"
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace llama_cpp {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = exec_aten::RuntimeContext;
using Error = torch::executor::Error;

// Helper function to create a ggml tensor with preallocated memory
static struct ggml_tensor * ggml_tensor_from(const Tensor & t, const int64_t * ne_override) {
    // HACK: since this is only used by mm, hardcode n_dims to 2
    // Should be t.dim() but that requires refactoring
    int n_dims = 2;
    // ET_CHECK_MSG(n_dims >= 1 && n_dims <= GGML_MAX_DIMS, "dimension %d is not within range (1, %d)", n_dims, GGML_MAX_DIMS);

    void * data = t.mutable_data_ptr();

    // TODO use memory from context to create tensor
    struct ggml_tensor * const result = (struct ggml_tensor *) malloc(sizeof (struct ggml_tensor));

    ET_CHECK_MSG(t.scalar_type() == exec_aten::ScalarType::Float, "only float type supported");
    // TODO support different types
    enum ggml_type type = ggml_type::GGML_TYPE_F32;
    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ GGML_BACKEND_CPU,
        /*.buffer       =*/ NULL,
        /*.n_dims       =*/ n_dims,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.is_param     =*/ false,
        /*.grad         =*/ NULL,
        /*.src          =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.view_src     =*/ NULL,
        /*.view_offs    =*/ 0,
        /*.data         =*/ data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //ggml_assert_aligned(result->data);

    if (ne_override != NULL) {
        for (int i = 0; i < n_dims; i++) {
            result->ne[i] = ne_override[i];
        }
    } else {
        for (int i = 0; i < n_dims; i++) {
            result->ne[i] = t.sizes()[i];
        }
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    // ctx->n_objects++;

    return result;
}

// View(mat2, {1, 64}), transpose, then matmul.
Tensor&
mm_out(RuntimeContext& ctx, const Tensor& in, const Tensor& mat2, Tensor& out) {

  // prepare input tensors
  // HACK: view(mat2, {64, 1});
  const int64_t dims[4] = {64, 1, 1, 1};

  struct ggml_tensor * a = ggml_tensor_from(in, NULL);

  struct ggml_tensor * b = ggml_tensor_from(mat2, dims);

//   GGML_ASSERT(ggml_can_mul_mat(b, a));
//   GGML_ASSERT(!ggml_is_transposed(b));

  const int64_t ne[4] = { b->ne[1], a->ne[1], a->ne[2], a->ne[3] };
  struct ggml_tensor * result = ggml_tensor_from(out, ne);

  result->op   = GGML_OP_MUL_MAT;
  result->grad = NULL;
  result->src[0] = b;
  result->src[1] = a;

  // run op
  struct ggml_cgraph gf = ggml_build_forward(result);

  struct ggml_cplan plan = ggml_graph_plan(&gf, /*int n_threads*/1);
  int res = ggml_graph_compute(&gf, &plan);

  return out;
}

} // namespace native
} // namespace llama_cpp
