//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include "hvx_funcs.h"

#ifndef PREPARE_DISABLED
API_EXPORT QuickShape simpledim_chunk1_4d_split_start(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
{
  size_t dims[4] = { 0, 0, 0, 0 };
  dims[dim] = splitinfo.start / splitinfo.size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT QuickShape simpledim_chunk1_4d_split_size(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
{
  size_t dims[4] = {
    orig.dim(rpx.graph(), 0),
    orig.dim(rpx.graph(), 1),
    orig.dim(rpx.graph(), 2),
    orig.dim(rpx.graph(), 3)
  };
  dims[dim] = 1;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}
#endif

BEGIN_PKG_OP_DEFINITION(PKG_TMANLinear);

static Qnn_Scalar_t sg_opDefaultGroup_SizeScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                   .int32Value = 64};
static Qnn_Param_t sg_opDefaultGroup_Size = {.paramType = QNN_PARAMTYPE_SCALAR,
                                            .scalarParam = sg_opDefaultGroup_SizeScalar};
static Qnn_Scalar_t sg_opDefaultBitsScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                             .int32Value = 2};
static Qnn_Param_t sg_opDefaultBits = {.paramType = QNN_PARAMTYPE_SCALAR,
                                      .scalarParam = sg_opDefaultBitsScalar};
static Qnn_Scalar_t sg_opDefaultSymmetricScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                  .int32Value = 0};
static Qnn_Param_t sg_opDefaultSymmetric = {.paramType = QNN_PARAMTYPE_SCALAR,
                                           .scalarParam = sg_opDefaultSymmetricScalar};

template<typename TensorType>
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric);

static float tmanlinearCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanlinearImpl<Tensor>), "TMANLinear")

DEF_TENSOR_PROPERTIES(
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  Flat("*", "qweight", "scales"),
  MainMemory("qweight", "scales", "group_size", "bits", "symmetric"),
  Tcm("*", "l"))

#define SIZE_OF(WEIGHT) MUL(ELEMENTSIZE_OF(WEIGHT), DIM_OF(WEIGHT, 0), DIM_OF(WEIGHT, 1), DIM_OF(WEIGHT, 2), DIM_OF(WEIGHT, 3))

// GPTQ
DEF_PACKAGE_OPTIMIZATION(
  EARLY,
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  AND(GT(DIM_OF("qweight", 2), 1), GT(SIZE_OF("scales"), 128)),
  AUTOSPLIT(3, "I", DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
    Op(
      "TMANLinear", "l",
      AUTOSPLIT_SLICE("qweight",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "qweight", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "qweight", 2)),
      AUTOSPLIT_SLICE("scales",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "scales", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "scales", 2)),
      "group_size", "bits", "symmetric")))

// BitNet: weight scale shouldn't be split
DEF_PACKAGE_OPTIMIZATION(
  EARLY + 1,
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  AND(GT(DIM_OF("qweight", 2), 1), LE(SIZE_OF("scales"), 128)),
  AUTOSPLIT(3, "I", DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
    Op(
      "TMANLinear", "l",
      AUTOSPLIT_SLICE("qweight",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "qweight", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "qweight", 2)),
      "scales", "group_size", "bits", "symmetric")))

DEF_PACKAGE_PARAM_ORDER("TMANLinear",
                        "group_size",
                        false,
                        &sg_opDefaultGroup_Size,
                        "bits",
                        false,
                        &sg_opDefaultBits,
                        "symmetric",
                        false,
                        &sg_opDefaultSymmetric)

template<typename TensorType>
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric)
{
  using LType = int16_t;
  using XType = __fp16;
  using CType = float;

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G          = 4;
  constexpr int32_t LUT_SIZE       = 16;
  constexpr int32_t TILE_K         = 256;

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits       = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool zero_point    = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;

  const int32_t gemm_n = c.dims()[2];
  const int32_t gemm_m = c.dims()[3] / sizeof(float) / bits;
  const int32_t gemm_k = qweight.dims()[2] * qweight.dims()[3] * 32 / bits / gemm_m;

  const int32_t l_size  = gemm_k / LUT_G * LUT_SIZE;
  const int32_t ls_size = (ACT_GROUP_SIZE == -1) ? 1 : (gemm_k / ACT_GROUP_SIZE);

  const LType* l_ptr  = (const LType*)l.raw_data_const();
  const float* ls_ptr = (const float*)(l_ptr + l_size);
  const float* lb_ptr = ls_ptr + MAX(ls_size, 128 / sizeof(float));

  const uint8_t* w_ptr = (const uint8_t*)qweight.raw_data_const();
  const XType* s_ptr   = (const XType*)scales.raw_data_const();
  CType* c_ptr         = (CType*)c.raw_data();

  if (zero_point && bits == 2 && group_size == 64)  // w2g64, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 2, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (!zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=True
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, true, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 64)  // w4g64, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (!zero_point && bits == 2 && group_size == 0)  // bitnet
  {
    hvx_tbl<LType, XType, CType, -1, 0, false, 2, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlinearCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinear);
