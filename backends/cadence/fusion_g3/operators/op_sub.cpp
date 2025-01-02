/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <xa_nnlib_kernels_api.h>
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::KernelRuntimeContext;
using executorch::runtime::canCast;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace G3 { 
namespace native {

#define XT_KERNEL_CHECK(ctx, out, kernel, ...) \
  const auto ret = kernel(__VA_ARGS__);        \
  ET_KERNEL_CHECK_MSG(                         \
      ctx,                                     \
      ret == 0,                                \
      InvalidArgument,                         \
      out,                                     \
      "Failed to run kernel: " #kernel "(" #__VA_ARGS__ ")");

Tensor& sub_out(KernelRuntimeContext& ctx,
                const Tensor& a,
                const Tensor& b,
                const Scalar& alpha,
                Tensor& out)
{

    // Common Dtype
    ScalarType common_type = executorch::runtime::promoteTypes(a.scalar_type(), b.scalar_type());
#ifdef OP_ARG_CHECK
    ScalarType alpha_type = torch::executor::native::utils::get_scalar_dtype(alpha);

    // Check alpha type
    ET_KERNEL_CHECK(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    // Check Common Dtype
    ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       canCast(alpha_type, common_type)),
      InvalidArgument,
      out);

    // Check Dim Order
    ET_KERNEL_CHECK(
      ctx,  executorch::runtime::tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

    // Resize
    ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);
#endif

    // Compute Dtype
    ScalarType compute_type = torch::executor::native::utils::get_compute_type(common_type);

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "sub.out";

    int kTensorDimensionLimit = 5;

    int inp1_shape[kTensorDimensionLimit];
    int inp2_shape[kTensorDimensionLimit];
    int out_shape[kTensorDimensionLimit];

    bool broadcast = 0;

    int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
    max_dim = out.dim() > max_dim ? out.dim() : max_dim;
  
    bool optimized = 1;
  
    for (int i = 0; i < max_dim; i++) {
      out_shape[i]  = 1;
      inp1_shape[i] = 1;
      inp2_shape[i] = 1;
    }
    
    int offset_out  = max_dim - out.dim();
    int offset_inp1 = max_dim - a.dim();
    int offset_inp2 = max_dim - b.dim();
    
    for (int i = 0; i < out.dim(); i++) {
      out_shape[i + offset_out] = out.size(i);
    }
    for (int i = 0; i < a.dim(); i++) {
      inp1_shape[i + offset_inp1] = a.size(i);
    }
    for (int i = 0; i < b.dim(); i++) {
    inp2_shape[i + offset_inp2] = b.size(i); 
    }
  
    /*find broadcast*/
    for (int i = 0; i < out.dim(); i++) {
          if (((inp1_shape[i]) != (out_shape[i])) || ((inp2_shape[i]) != (out_shape[i]))) {
  		broadcast = 1;
  	}
    }
  
    if ((broadcast == 1) && (max_dim > kTensorDimensionLimit)) {
      optimized = 0;
    }
    
    if ((compute_type == ScalarType::Int) && (optimized)){
      const int* const inp1_data = a.const_data_ptr<int>();
      const int* const inp2_data = b.const_data_ptr<int>();
      int* const out_data = out.mutable_data_ptr<int>();
  
      int alpha_val;
      torch::executor::native::utils::extract_scalar(alpha, &alpha_val);
  
      if (b.numel() == 1)
      {
        XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_sub_scalar_32x32_32,
          out_data,
          inp1_data,
          inp2_data[0],
          alpha_val, 
		  out.numel());
      }
      else if(broadcast)
      {
          XT_KERNEL_CHECK(
            ctx,
            out,
            xa_nn_elm_sub_broadcast_5D_32x32_32,
            out_data,
            out_shape,
            inp1_data,
            inp1_shape,
            inp2_data,
            inp2_shape,
            max_dim,
            alpha_val);
      }
      else
      {
           XT_KERNEL_CHECK(
             ctx,
             out,
             xa_nn_elm_sub_32x32_32,
             out_data,
             inp1_data,
             inp2_data,
             alpha_val,
             out.numel());
      }
    }
    else if((compute_type == ScalarType::Float) && (optimized))
    {
        const float* const inp1_data = a.const_data_ptr<float>();
        const float* const inp2_data = b.const_data_ptr<float>();
        float* const out_data = out.mutable_data_ptr<float>();

        float alpha_val;
        torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

        if (b.numel() == 1)
        {
          XT_KERNEL_CHECK(
            ctx,
            out,
            xa_nn_elm_sub_scalar_f32xf32_f32,
            out_data,
            inp1_data,
            inp2_data[0],
            alpha_val,
            out.numel());
        }
        else if(broadcast)
        {
            XT_KERNEL_CHECK(
              ctx,
              out,
              xa_nn_elm_sub_broadcast_5D_f32xf32_f32,
              out_data,
              out_shape,
              inp1_data,
              inp1_shape,
              inp2_data,
              inp2_shape,
              max_dim,
              alpha_val);
        }
        else
        {
            XT_KERNEL_CHECK(
              ctx,
              out,
              xa_nn_elm_sub_f32xf32_f32,
              out_data,
              inp1_data,
              inp2_data,
              alpha_val,
              out.numel());
        }
    }
    else
    {
        ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
            const CTYPE_COMPUTE val_alpha = torch::executor::native::utils::
            scalar_to<CTYPE_COMPUTE>(alpha);
            torch::executor::native::utils::
                apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
                [val_alpha](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
                  return val_a - val_alpha * val_b;
                },
                ctx,
                a,
                torch::executor::native::utils::SupportedTensorDtypes::REALHBF16,
                b,
                torch::executor::native::utils::SupportedTensorDtypes::REALHBF16,
                out,
                torch::executor::native::utils::SupportedTensorDtypes::REALHBF16);
        });
    }

    return out;
}

Tensor& sub_scalar_out(KernelRuntimeContext& ctx,
                       const Tensor& a,
                       const Scalar& b,
                       const Scalar& alpha,
                       Tensor& out)
{
    // Common Dtype
    ScalarType common_type = torch::executor::native::utils::promote_type_with_scalar(a.scalar_type(), b);
#ifdef OP_ARG_CHECK
    ScalarType alpha_type = torch::executor::native::utils::get_scalar_dtype(alpha);

    // Check alpha type
    ET_KERNEL_CHECK(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    // Check Common Dtype
    ET_KERNEL_CHECK(
      ctx,
      (common_type == out.scalar_type() && canCast(alpha_type, common_type)),
      InvalidArgument,
      out);

    // Check Dim Order
    ET_KERNEL_CHECK(
      ctx,  executorch::runtime::tensors_have_same_dim_order(a, out),
                       InvalidArgument, out);

    // Resize
    ET_KERNEL_CHECK(
      ctx, executorch::runtime::resize_tensor(out, a.sizes()) == Error::Ok,
           InvalidArgument, out);     
#endif

    // Compute Dtype
    ScalarType compute_type = torch::executor::native::utils::get_compute_type(common_type);

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "sub.Scalar_out";

    if(compute_type == ScalarType::Int)
    {
        const int* const inp1_data = a.const_data_ptr<int>();
        int inp2_val;
        torch::executor::native::utils::extract_scalar(b, &inp2_val);

        int alpha_val;
        torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

        int* const out_data = out.mutable_data_ptr<int>();

        XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_sub_scalar_32x32_32,
          out_data,
          inp1_data,
          inp2_val,
          alpha_val,
          out.numel());
    }
    else if(compute_type == ScalarType::Float)
    {
        const float* const inp1_data = a.const_data_ptr<float>();
        float inp2_val;
        torch::executor::native::utils::extract_scalar(b, &inp2_val);

        float alpha_val;
        torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

        float* const out_data = out.mutable_data_ptr<float>();

        XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_sub_scalar_f32xf32_f32,
          out_data,
          inp1_data,
          inp2_val,
          alpha_val,
          out.numel());
    }
    else
    {
        ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
            const CTYPE_COMPUTE val_b = torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(b);
            const CTYPE_COMPUTE val_alpha = torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(alpha);
            torch::executor::native::utils::apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
                [val_b, val_alpha](const CTYPE_COMPUTE val_a) {
                  return val_a - val_alpha * val_b;
                },
                ctx,
                a,
                torch::executor::native::utils::SupportedTensorDtypes::REALHBF16,
                out,
                torch::executor::native::utils::SupportedTensorDtypes::SAME_AS_COMMON);
        });
    }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence