#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/core/portable_type/tensor.h>  // for torch::executor::Tensor
#include <executorch/runtime/core/portable_type/scalar.h>  // for torch::executor::Scalar

extern "C" {
#include "Include/arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using Tensor = torch::executor::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

//__attribute__((section(".text_ddr")))
void softmax_wrapper(
    const int8_t* input_data,
    int rows,
    int cols,
    int32_t input_mult,
    int32_t input_shift,
    int32_t diff_min,
    int8_t* output_data) {
      arm_softmax_s8(
        input_data,
        rows,
        cols,
        input_mult,
        input_shift,
        diff_min,
        output_data);
}

torch::executor::Tensor& aten_softmax(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {

  ET_LOG(Info, "CMSIS-NN softmax kernel called");
  const int8_t* input_data = self.data_ptr<int8_t>();
  int8_t* output_data = out.data_ptr<int8_t>();

  int rows = self.sizes()[0];
  int cols = self.sizes()[1];
  ET_LOG(Info, "Input shape: %d x %d", rows, cols);
  // Quantization params - dummy values for now, refine later
  int32_t input_mult = 1 << 4;  // or something from qparams
  int32_t input_shift = 0;
  int32_t diff_min = -128;

  softmax_wrapper(
    input_data,
    rows,
    cols,
    input_mult,
    input_shift,
    diff_min,
    output_data);

  return out;
}

} // namespace native
} // namespace cortex_m
