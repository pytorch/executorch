//=============================================================================
//
//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All rights reserved
//
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree.
//
//============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_Embedding);

// op execute function declarations
template <typename TensorType, typename TensorType1>
int embeddingImpl(TensorType &output, const TensorType1 &input, const Int32Tensor_TCM &indices);

/*
 * method for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 */
DEF_PACKAGE_OP(
  (embeddingImpl<QuantUint8Tensor_TCM, QuantUint8Tensor_TCM>),
  "Embedding"
)

DEF_TENSOR_PROPERTIES(
  Op("Embedding", "input", "indices"),
  Flat("input", "indices")
)

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */
DEF_PACKAGE_OPTIMIZATION(
  PRE_TRANSLATE,
  Op(
    "Embedding",
     "table",
     LET("InputOp",
       Op(FROM_DEFAULT_PACKAGE("*Input"), "indices", "original", "effective")
     )
  ),
  AND(
    IS_INT32("InputOp"),
    IS_SHAPE_1x1x1xd("InputOp"),
    EQ(DIM_HEIGHT("table"), 1),
    EQ(DIM_BATCHES("table"), 1),
    EQ(DIM_WIDTH("*"), DIM_DEPTH("InputOp")),
    EQ(DIM_DEPTH("*"), DIM_DEPTH("table")),
    SAME_SHAPE("original", "effective"),
    SAME_SHAPE("original", "InputOp"),
    NE(RANK_OF("InputOp"), 5),
    NE(RANK_OF("*"), 5),
    IS_QUINT8("table")
  ),
  Op(
    FROM_DEFAULT_PACKAGE("*InputGather2DDMAQuant"),
    "table",
    "indices",
    gen_Shape(1, 1, DIM_DEPTH("InputOp"), 1),
    gen_Shape(1, 1, DIM_DEPTH("InputOp"), 1),
    gen_Shape(0, 0, 0, 0),
    gen_ShapeOf("*"),
    gen_Shape(0, 0, 0, 0)
  )
)

/* execute functions for ops */
template <typename TensorType, typename TensorType1>
int embeddingImpl(TensorType &output, const TensorType1 &input, const Int32Tensor_TCM &indices) {
  // NOTE: this implementation is not intended to be used
  //       op should be replaced with HTP optimized version
  // input dim: (1, 1, n, d)
  // indices dim: (1, m, k, l)
  // output dim: (m, k, l, d)
  size_t out_dims[4] = {indices.dim(1), indices.dim(2), indices.dim(3), input.dim(3)};
  output.set_dims(out_dims);
  uint8_t* out_ptr = (uint8_t *)output.get_raw_addr(0, 0, 0, 0);
  const uint8_t* in_base = (const uint8_t*)input.get_raw_addr(0, 0, 0, 0);
  const int32_t* indices_base = (const int32_t*)indices.get_raw_addr(0, 0, 0, 0);
  size_t indices_len = 1, input_stride = input.dim(3);
  for (int i = 0; i < 4; ++i) {
    indices_len *= indices.dim(i);
  }
  for (int i = 0; i < indices_len; ++i) {
    memcpy(out_ptr, in_base + indices_base[i]*input_stride, input_stride);
    out_ptr += input_stride;
  }
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_Embedding);
