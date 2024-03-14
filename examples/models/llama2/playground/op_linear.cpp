/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ggml-quants.h"
#include "ggml.h"
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <torch/torch.h>
#include <torch/library.h>

namespace llama_cpp {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = exec_aten::RuntimeContext;
using Error = torch::executor::Error;

static void ggml_compute_forward_mul_mat(
        const void * restrict src0,
        int64_t * ne0s,
        size_t * nb0s,
        const float * restrict src1,
        int64_t * ne1s,
        size_t * nb1s,
        float * restrict dst,
        int64_t * nes,
        size_t * nbs) {
    // Takes a q4_0 weight (src0) and a float activation (src1)

    // src0 dim, this is the weight
    int64_t ne00 = ne0s[0];
    int64_t ne01 = ne0s[1];
    int64_t ne02 = ne0s[2];
    int64_t ne03 = ne0s[3];

    size_t nb00 = nb0s[0];
    size_t nb01 = nb0s[1];
    size_t nb02 = nb0s[2];
    size_t nb03 = nb0s[3];

    // src1 dim, this is the activation
    int64_t ne10 = ne1s[0];
    int64_t ne11 = ne1s[1];
    int64_t ne12 = ne1s[2];
    int64_t ne13 = ne1s[3];

    size_t nb00 = nb0s[0];
    size_t nb01 = nb0s[1];
    size_t nb02 = nb0s[2];
    size_t nb03 = nb0s[3];
    // dst dim
    int64_t ne0 = nes[0];
    int64_t ne1 = nes[1];
    int64_t ne2 = nes[2];
    int64_t ne3 = nes[3];

    size_t nb0 = nbs[0];
    size_t nb1 = nbs[1];
    size_t nb2 = nbs[2];
    size_t nb3 = nbs[3];

    // single thread
    const int ith = 0;
    const int nth = 1;

    // const enum ggml_type type = src0->type;

    // const bool src1_cont = ggml_is_contiguous(src1);

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    // GGML_ASSERT(nb00 == ggml_type_size(type));
    // GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // quantize activation
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, ne10);
    char * buffer = (char *) malloc(ne11*ne12*ne13*row_size)
    char * wdata = buffer;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                quantize_row_q8_0((float *)((char *) src1 + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                wdata += row_size;
            }
        }
    }


    const void * wdata = buffer;

    const int64_t nr0 = ne01;          // src0 rows
    const int64_t nr1 = ne1*ne12*ne13; // src1 rows

    //printf("nr0 = %lld, nr1 = %lld\n", nr0, nr1);

    // distribute the thread work across the inner or outer loop based on which one is larger

    const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
    const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

    const int64_t ith0 = ith % nth0;
    const int64_t ith1 = ith / nth0;

    const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
    const int64_t dr1 = (nr1 + nth1 - 1)/nth1;

    const int64_t ir010 = dr0*ith0;
    const int64_t ir011 = MIN(ir010 + dr0, nr0);

    const int64_t ir110 = dr1*ith1;
    const int64_t ir111 = MIN(ir110 + dr1, nr1);

    //printf("ir010 = %6lld, ir011 = %6lld, ir110 = %6lld, ir111 = %6lld\n", ir010, ir011, ir110, ir111);

    // threads with no work simply yield (not sure if it helps)
    // if (ir010 >= ir011 || ir110 >= ir111) {
    //     sched_yield();
    //     return;
    // }

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t nrc = 1;
    // TODO: currently the mmla kernels support only even numbered rows/cols.
    // this check can be removed once they are extended to support odd numbered rows/cols too
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        nrc = 1;
    }

    const size_t src1_col_stride = row_size;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
        for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ir1 += nrc) {
                const int64_t i13 = (ir1/(ne12*ne1));
                const int64_t i12 = (ir1 - i13*ne12*ne1)/ne1;
                const int64_t i11 = (ir1 - i13*ne12*ne1 - i12*ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13/r3;
                const int64_t i02 = i12/r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char *) src0 + (0 + i02*nb02 + i03*nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata + (i11+i12*ne11 + i13*ne12*ne11)*row_size;
                float * dst_col = (float *) ((char *) dst + (i1*nb1 + i2*nb2 + i3*nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ir0 += nrc) {
                    ggml_vec_dot_q4_0_q8_0(ne00, &tmp[ir0 - iir0], (nrc>1 ? 16 : 0), src0_row + ir0*nb01, (nrc>1 ? nb01 : 0), src1_col, (nrc>1 ? src1_col_stride : 0), nrc);
                }

                for (int cn = 0; cn < nrc; ++cn) {
                    memcpy(&dst_col[iir0 + cn*nb1/nb0], tmp + (cn*16), (MIN(iir0 + blck_0, ir011) - iir0)*sizeof(float));
                }
            }
        }
    }
    free(buffer);
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
linear_q4_0_out(const Tensor& weights, const Tensor& scale, const Tensor& activation, Tensor& out) {
  // weights are int4 quantized values stored in int8 tensors, i.e., first 4 bits are 0.
  // scale contains scales for groupwise (32) quantized values, numel = weights.numel() / 32
  // activation and out are float32 tensor
  const void * weights_packed = pack_q4_0(weights, scale);
  int64_t weights_sizes[4];
  for (int i = 0; i < 4; i++) {
    weights_sizes[i] = weights.size(i);
  }
  size_t weights_byte_sizes[4]; // strides * sizeof(block_q4_0)
  weights_byte_sizes[0] = sizeof(block_q4_0);
  for (int i = 1; i < 4; i++) {
    weights_byte_sizes[i] = weights.size(i-1) / QK4_0 * weights_byte_sizes[i-1];
  }
  // activation
  const float * input = activation.const_data_ptr<float>();
  int64_t input_sizes[4];
  for (int i = 0; i < 4; i++) {
    input_sizes[i] = activation.size(i);
  }
  size_t input_byte_sizes[4];
  input_byte_sizes[0] = sizeof(float);
  for (int i = 1; i < 4; i++) {
    input_byte_sizes[i] = activation.size(i-1) * input_byte_sizes[i-1];
  }
  // out
  float * out_data = out.mutable_data_ptr<float>();
  int64_t out_sizes[4];
  for (int i = 0; i < 4; i++) {
    out_sizes[i] = out.size(i);
  }
  size_t out_byte_sizes[4];
  out_byte_sizes[0] = sizeof(float);
  for (int i = 1; i < 4; i++) {
    out_byte_sizes[i] = out.size(i-1) * out_byte_sizes[i-1];
  }

  ggml_compute_forward_mul_mat(weights_packed, &weights_sizes, &weights_byte_sizes, input, &input_sizes, &input_byte_sizes, out_data, &out_sizes, &out_byte_sizes);

  free(weights_packed);
  return out;
}

Tensor&
linear_q4_0_out_with_context(RuntimeContext& context, const Tensor& weights, const Tensor& scale, const Tensor& activation, Tensor& out) {
    (void)context;
    return linear_q4_0_out(weights, scale, zeros, activation, out);
}


static Kernel k = Kernel::make_boxed_kernel("ggml::linear_q4_0.out", EXECUTORCH_FN(linear_q4_0_out));
auto a = register_kernels({k});

at::Tensor linear_q4_0(const at::Tensor& weight, const at::Tensor& scale, const at::Tensor& input) {
  auto output = at::empty({input.size(0), weight.size(0)}, input.options().dtype(at::kHalf));
  WRAP(linear_q4_0_out, 4)((weight, scale, input, output));
  return output;
}

} // namespace native
} // namespace llama_cpp

TORCH_LIBRARY(ggml, m) {
    m.def("linear_q4_0(Tensor weight, Tensor scale, Tensor input) -> Tensor", &llama_cpp::native::linear_q4_0);
    m.def("linear_q4_0.out(Tensor weight, Tensor scale, Tensor input, *, Tensor(a!) out) -> Tensor(a!)", WRAP(llama_cpp::native::linear_q4_0_out, 4));
}
