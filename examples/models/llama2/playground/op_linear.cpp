/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <torch/torch.h>
#include <torch/library.h>

namespace llama_cpp {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = exec_aten::RuntimeContext;
using Error = torch::executor::Error;

// Helper function to create a ggml tensor with preallocated memory
static struct ggml_tensor * ggml_tensor_from(const Tensor & t, const int64_t * ne_override, const char* name) {
    // Modified from ggml_new_tensor_impl()
    int n_dims = t.dim();
    ET_CHECK_MSG(n_dims >= 1 && n_dims <= GGML_MAX_DIMS, "dimension %d is not within range (1, %d)", n_dims, GGML_MAX_DIMS);

    enum ggml_type type;
    switch (t.scalar_type()) {
        case exec_aten::ScalarType::Byte:
            type = GGML_TYPE_Q4_0; // hardcoded
            break;
        case exec_aten::ScalarType::Half:
            type = GGML_TYPE_F16;
            break;
        case exec_aten::ScalarType::Float:
            type = GGML_TYPE_F32;
            break;
        default:
            ET_CHECK_MSG(false, "unsupported scalar type %hdd", t.scalar_type());
    }

    // TODO use memory from context to create tensor
    struct ggml_tensor * const result = (struct ggml_tensor *) malloc(sizeof (struct ggml_tensor));
    void * data = t.mutable_data_ptr();

    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ GGML_BACKEND_CPU,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
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

    int i = 0;
    while (*(name + i) != '\0') {
        result->name[i] = *(name + i++);
    }

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


// Helper function to create a ggml q4_0 tensor with preallocated memory
static void * pack_q4_0(const Tensor & t, const Tensor & scale) {
    int n_dims = t.dim();
    ET_CHECK_MSG(n_dims >= 1 && n_dims <= GGML_MAX_DIMS, "dimension %d is not within range (1, %d)", n_dims, GGML_MAX_DIMS);

    enum ggml_type type = GGML_TYPE_Q4_0;

    // TODO use memory from context to create tensor
    struct ggml_tensor * const result = (struct ggml_tensor *) malloc(sizeof (struct ggml_tensor));
    ET_CHECK_MSG(t.scalar_type() == exec_aten::ScalarType::Byte, "Expected t to be Byte tensor but got %hdd", t.scalar_type());
    ET_CHECK_MSG(scale.scalar_type() == exec_aten::ScalarType::Float, "Expected scale to be Float tensor but got %hdd", scale.scalar_type());

    // prepare a temp buffer to store the packed quantized values. Each block_q4_0 contains half of the group size (32 / 2 = 16) of uint8_t values and a fp16 scale value.
    ET_CHECK_MSG(t.numel() % QK4_0 == 0, "Expecting numel to be multiple of %d but got %zu", QK4_0, t.numel());
    static const int qk = QK4_0;

    size_t group_num = t.numel() / qk;
    block_q4_0 buf[group_num];
    int8_t* data = t.mutable_data_ptr<int8_t>();
    float* scales = scale.mutable_data_ptr<float>();

    // data here is int8 unpacked quantized values, need to convert to packed int4 format
    for (size_t i = 0; i < group_num; ++i) {
        int8_t* group_start = data + i * qk;
        int8_t* group_end = data + (i+1) * qk;
        block_q4_0* block = buf + i;

        block->scale = GGML_FP32_TO_FP16(scales[i]);
        for (int j = 0; j < QK4_0/2; ++j) {
            block->qs[j]  = group_start[j];
            block->qs[j] |= group_start[qk/2 + j] << 4;
        }
    }

    // memcopy the packed data into a new data from heap. This is safe because sizeof(block_q4_0) * group_num is smaller than t.numel()
    void * dest = malloc(sizeof(block_q4_0) * group_num);
    memcpy(dest, buf, sizeof(block_q4_0) * group_num);
    return dest;
}

Tensor&
linear_q4_0_out(const Tensor& weights, const Tensor& scale, const Tensor& zeros, const Tensor& activation, Tensor& out) {
  (void)zeros; // ggml hardcode all zeros to be 8.5 for Q4_0 quantization

  struct ggml_tensor * a = ggml_tensor_from(weights, NULL, "a");
  void* packed_data = pack_q4_0(weights, scale); // alloc memory under the hood
  a->data = packed_data;

  struct ggml_tensor * b = ggml_tensor_from(activation, NULL, "b");

//   GGML_ASSERT(ggml_can_mul_mat(b, a));
//   GGML_ASSERT(!ggml_is_transposed(b));

  const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
  struct ggml_tensor * result = ggml_tensor_from(out, ne, "result");

  result->op   = GGML_OP_MUL_MAT;
  result->grad = NULL;
  result->src[0] = a;
  result->src[1] = b;

  // run op
  struct ggml_cgraph gf = ggml_build_forward(result);

  struct ggml_cplan plan = ggml_graph_plan(&gf, /*int n_threads*/1);

  int res = ggml_graph_compute(&gf, &plan);

  free(packed_data);
  free(a);
  free(b);
  free(result);
  return out;
}

Tensor&
linear_q4_0_out_with_context(RuntimeContext& context, const Tensor& weights, const Tensor& scale, const Tensor& zeros, const Tensor& activation, Tensor& out) {
    (void)context;
    return linear_q4_0_out(weights, scale, zeros, activation, out);
}


static Kernel k = Kernel::make_boxed_kernel("ggml::linear_q4_0.out", EXECUTORCH_FN(linear_q4_0_out));
auto a = register_kernels({k});

at::Tensor linear_q4_0(const at::Tensor& weight, const at::Tensor& scale, const at::Tensor & zero_point, const at::Tensor& input) {
  auto output = at::empty({input.size(0), weight.size(0)}, input.options().dtype(at::kHalf));
  WRAP(linear_q4_0_out, 4)((weight, scale, zero_point, input, output));
  return output;
}

} // namespace native
} // namespace llama_cpp

TORCH_LIBRARY(ggml, m) {
    m.def("linear_q4_0(Tensor weight, Tensor scale, Tensor zeros, Tensor input) -> Tensor", &llama_cpp::native::linear_q4_0);
    m.def("linear_q4_0.out(Tensor weight, Tensor scale, Tensor zeros, Tensor input, *, Tensor(a!) out) -> Tensor(a!)", WRAP(llama_cpp::native::linear_q4_0_out, 4));
}
