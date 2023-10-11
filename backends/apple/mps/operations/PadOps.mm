//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"
#include <ATen/native/mps/OperationUtils.h>

namespace mps {
using namespace torch;

// Pad operations (1D/2D/3D forward)
static PyMPSGraphTensor*
pad_out_template(MPSGraph* mpsGraph,
                 MPSGraphTensor* input, IntArrayRef padding,
                 MPSGraphPaddingMode mode, double constantValue)
{
  const int padding_size = (int) padding.size();
  int padding_dim = padding_size / 2; // either 1D, 2D, or 3D

  TORCH_CHECK(padding_size == 2 || padding_size == 4 || padding_size == 6,
              "invalid padding argument of size ", padding_size);

  auto input_sizes = getMPSShapeVec(input.shape);
  int64_t nbatch = 1;
  int64_t ndims = input_sizes.size();

  TORCH_CHECK(ndims >= (int64_t)padding_dim, "Length of pad should be no more than twice the number of "
              "dimensions of the input. Pad length is ", padding_size, "while the input has ", ndims, "dimensions.");

  // number of input dims with ConstantPad could be less than 2
  int dim_w = padding_dim;
  int dim_h = padding_dim - 1;
  int dim_d = padding_dim - 2;
  int dim_slices = 0;

  if (mode != MPSGraphPaddingModeConstant && ndims > padding_dim) {
    bool valid_dims = input_sizes[1] != 0 && input_sizes[padding_dim] != 0;
    TORCH_CHECK((ndims == 1 + padding_dim && valid_dims) ||
                (ndims == 2 + padding_dim && valid_dims && input_sizes[1 + padding_dim] != 0),
                "3D or 4D (batch mode) tensor expected for input, but got: ", input);
  }

  if (ndims == padding_dim) {
    dim_w--;
    dim_h--;
    dim_d--;
  } else if (ndims > padding_dim + 1) {
    const int dim_diff = (int)ndims - padding_dim - 1;
    // this virtually inflates the padding with zeros if ndims > padding_dim + 2
    padding_dim += dim_diff - 1;
    dim_w += dim_diff;
    dim_h += dim_diff;
    dim_d += dim_diff;
    dim_slices++;
    nbatch = input_sizes[0];
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding_size > 2 ? padding[2] : 0;
  int64_t pad_b = padding_size > 2 ? padding[3] : 0;
  int64_t pad_front = padding_size > 4 ? padding[4] : 0;
  int64_t pad_back  = padding_size > 4 ? padding[5] : 0;

  int64_t nplane = input_sizes[dim_slices];
  int64_t input_w = input_sizes[dim_w];
  int64_t output_w  = input_w + pad_l + pad_r;
  int64_t input_h = padding_dim > 1 ? input_sizes[dim_h] : 0;
  int64_t output_h = padding_dim > 1 ? input_h + pad_t + pad_b : 0;
  int64_t input_d = padding_dim > 2 ? input_sizes[dim_d] : 0;
  int64_t output_d = padding_dim > 2 ? input_d + pad_front + pad_back : 0;

  TORCH_CHECK(output_w >= 1 || output_h >= padding_dim - 1,
    "input (H: ", input_h, ", W: ", input_w, ") is too small. Calculated "
    "output H: ", output_h, " W: ", output_w);

  // these checks are only relevant for reflection padding (code taken from ReflectionPad.cpp)
  if (mode == MPSGraphPaddingModeReflect) {
    TORCH_CHECK(pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size should be less than the corresponding "
      "input dimension, but got: padding (", pad_l, ", ", pad_r,
      ") at dimension ", dim_w, " of input ", ndims);

    if (padding_dim > 1) {
      TORCH_CHECK(pad_t < input_h && pad_b < input_h,
        "Argument #6: Padding size should be less than the corresponding "
        "input dimension, but got: padding (", pad_t, ", ", pad_b,
        ") at dimension ", dim_h, " of input ", ndims);
    }
    if (padding_dim > 2) {
      TORCH_CHECK(pad_front < input_d && pad_back < input_d,
        "Argument #8: Padding size should be less than the corresponding "
        "input dimension, but got: padding (", pad_front, ", ", pad_back,
        ") at dimension ", dim_d, " of input ", ndims);
    }
  }

  std::vector<NSNumber*> leftPadVec(ndims, @(0));
  std::vector<NSNumber*> rightPadVec(ndims, @(0));

  for (int64_t pdim = 0; pdim < padding_size / 2; pdim++) {
    const int64_t leftIdx  = pdim * 2;
    const int64_t rightIdx = pdim * 2 + 1;
    const int64_t padIdx = ndims - pdim - 1;

    leftPadVec [padIdx] = @(padding[leftIdx]);
    rightPadVec[padIdx] = @(padding[rightIdx]);
  }
  MPSShape *leftPadding  = [NSArray arrayWithObjects:leftPadVec.data() count:ndims];
  MPSShape *rightPadding = [NSArray arrayWithObjects:rightPadVec.data() count:ndims];

  // TODO: check if Bool type works with Constant padding (asserts on pytorch)
  MPSGraphTensor *padTensor = [mpsGraph padTensor: input
                                  withPaddingMode: mode
                                      leftPadding: leftPadding
                                     rightPadding: rightPadding
                                    constantValue: constantValue
                                             name: nil];

  return padTensor;
}

PyMPSGraphTensor*
MPSGraphModule::constant_pad_nd(
    MPSGraphTensor* input,
    IntArrayRef pad,
    const double value) {
  return pad_out_template(mpsGraph, input, pad, MPSGraphPaddingModeConstant, value);
}

} // namespace at::native
