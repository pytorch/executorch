//==============================================================================
// Auto Generated Code for FastGeluOpPackage
//==============================================================================

#include <algorithm>
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_FastGelu);

// op execute function declarations
template <typename TensorType>
GraphStatus fastgeluImpl(TensorType& y, const TensorType& x);

// forward declaration of sample cost function
static float fastgeluCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default
 * flag (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((fastgeluImpl<Tensor>), "FastGelu")
 */
DEF_PACKAGE_OP((fastgeluImpl<Tensor>), "FastGelu")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or more flags,
 * FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX, RESOURCE_HMX(not
 * supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((fastgeluImpl<PlainFloatTensor>),
 * "FastGelu", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((fastgeluImpl<PlainFloatTensor>),
 * "FastGelu", fastgeluCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core
 * documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax:
 * DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
 * order of parameters listed determines the order of parameters passed into op
 * execution functions if an op does not have a parameter order definition,
 * parameter order passed into Qnn_addNode will be passed into op execution
 * functions if an op has a parameter order definition, any parameter passed
 * into Qnn_addNode with unlisted name will be abandoned if two or more op
 * packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at
 * Qnn_addNode DEFAULT is used when MANDATORY is false if provided as
 * Qnn_Param_t*, DEFAULT will be used for graph construction when this parameter
 * is not provided at Qnn_addNode if provided as nullptr, graph construction
 * will skip this parameter when this parameter is not provided at Qnn_addNode
 */

/* execute functions for ops */

// template <typename TensorType>
// GraphStatus fastgeluImpl(TensorType& y, const TensorType& x) {
//   const uint32_t numElements = x.total_storage_elements();

//   if (y.total_storage_elements() != numElements) {
//     return GraphStatus::ErrorFatal;
//   }

//   const float kAlpha = 0.7978845608f; // sqrt(2/pi)
//   const float kCoeff = 0.044715f;

//   float* yData = reinterpret_cast<float*>(y.raw_data());
//   const float* xData = reinterpret_cast<const float*>(x.raw_data_const());

//   for (uint32_t i = 0; i < numElements; ++i) {
//     const float v = xData[i];
//     const float inner = kAlpha * (v + kCoeff * v * v * v);
//     yData[i] = 0.5f * v * (1.0f + std::tanh(inner));
//   }

//   return GraphStatus::Success;
// }

template <typename TensorType>
GraphStatus fastgeluImpl(TensorType& y, const TensorType& x) {
  const uint32_t N = x.total_storage_elements();

  if (y.total_storage_elements() != N) {
    return GraphStatus::ErrorFatal;
  }

  const auto in_info = x.get_dtype_intfc();
  const auto out_info = y.get_dtype_intfc();

  if (in_info.dtype != DType::Float32 || in_info.dtype != DType::QUInt8) {
    return GraphStatus::ErrorPrecision;
  }
  if (in_info.dtype == DType::Float32 && out_info.dtype == DType::Float32) {
    const float* xData = static_cast<const float*>(x.raw_data_const());
    float* yData = static_cast<float*>(y.raw_data());

    // --- Temporary FP16 buffers ---
    std::vector<Float16> tmp_in(N);
    std::vector<Float16> tmp_out(N);

    for (uint32_t i = 0; i < N; ++i) {
      tmp_in[i] = static_cast<Float16>(xData[i]);
    }

#ifdef __hexagon__
    union {
      Float16 f;
      uint16_t b;
    } kAlpha = {(Float16)0.7978845608f}; // sqrt(2/pi)
    union {
      Float16 f;
      uint16_t b;
    } kCoeff = {(Float16)0.044715f};
    union {
      Float16 f;
      uint16_t b;
    } kHalf = {(Float16)0.5f};
    union {
      Float16 f;
      uint16_t b;
    } kOne = {(Float16)1.0f};
    union {
      Float16 f;
      uint16_t b;
    } k27 = {(Float16)27.0f};
    union {
      Float16 f;
      uint16_t b;
    } kInv27 = {(Float16)(1.0f / 27.0f)};
    union {
      Float16 f;
      uint16_t b;
    } kOne3 = {(Float16)(1.0f / 3.0f)};
    union {
      Float16 f;
      uint16_t b;
    } kOne9 = {(Float16)(1.0f / 9.0f)};

    HVX_Vector v_alpha = Q6_Vh_vsplat_R(kAlpha.b);
    HVX_Vector v_coeff = Q6_Vh_vsplat_R(kCoeff.b);
    HVX_Vector v_half = Q6_Vh_vsplat_R(kHalf.b);
    HVX_Vector v_one = Q6_Vh_vsplat_R(kOne.b);
    HVX_Vector v_27 = Q6_Vh_vsplat_R(k27.b);
    HVX_Vector v_inv27 = Q6_Vh_vsplat_R(kInv27.b);
    HVX_Vector v_1_3 = Q6_Vh_vsplat_R(kOne3.b);
    HVX_Vector v_1_9 = Q6_Vh_vsplat_R(kOne9.b);

    const int VBYTES = 128;
    const int ELEMS = VBYTES / sizeof(Float16); // 64

    for (uint32_t i = 0; i < N; i += ELEMS) {
      HVX_Vector vx = q6op_V_vldu_A(&tmp_in[i]); // x
      HVX_Vector vx2 = Q6_Vhf_vmpy_VhfVhf(vx, vx); // x^2
      HVX_Vector vx3 = Q6_Vhf_vmpy_VhfVhf(vx2, vx); // x^3

      // z = α * (x + c*x^3)
      HVX_Vector vcx3 = Q6_Vhf_vmpy_VhfVhf(vx3, v_coeff);
      HVX_Vector vsum = Q6_Vhf_vadd_VhfVhf(vx, vcx3);
      HVX_Vector vz = Q6_Vhf_vmpy_VhfVhf(vsum, v_alpha);

      // z^2, z^4
      HVX_Vector vz2 = Q6_Vhf_vmpy_VhfVhf(vz, vz);
      HVX_Vector vz4 = Q6_Vhf_vmpy_VhfVhf(vz2, vz2);

      // inv_den ≈ (1/27) * (1 - (1/3) z^2 + (1/9) z^4)
      HVX_Vector term1 = Q6_Vhf_vmpy_VhfVhf(vz2, v_1_3); // (1/3) z^2
      HVX_Vector one_m_t = Q6_Vhf_vsub_VhfVhf(v_one, term1); // 1 - (1/3) z^2
      HVX_Vector term2 = Q6_Vhf_vmpy_VhfVhf(vz4, v_1_9); // (1/9) z^4
      HVX_Vector poly =
          Q6_Vhf_vadd_VhfVhf(one_m_t, term2); // 1 - 1/3 z^2 + 1/9 z^4
      HVX_Vector inv_den = Q6_Vhf_vmpy_VhfVhf(poly, v_inv27); // * (1/27)

      // num = z * (27 + z^2) = 27z + z^3
      HVX_Vector z3 = Q6_Vhf_vmpy_VhfVhf(vz2, vz);
      HVX_Vector t27z = Q6_Vhf_vmpy_VhfVhf(vz, v_27);
      HVX_Vector num = Q6_Vhf_vadd_VhfVhf(t27z, z3);

      // tanh(z) ≈ num * inv_den
      HVX_Vector vtanh = Q6_Vhf_vmpy_VhfVhf(num, inv_den);

      // y = 0.5 * x * (1 + tanh)
      HVX_Vector one_plus_tanh = Q6_Vhf_vadd_VhfVhf(v_one, vtanh);
      HVX_Vector t = Q6_Vhf_vmpy_VhfVhf(vx, one_plus_tanh);
      HVX_Vector vy = Q6_Vhf_vmpy_VhfVhf(t, v_half);

      q6op_vstu_AV(&tmp_out[i], vy);
    }
#else
    // Scalar fallback
    for (uint32_t i = 0; i < N; ++i) {
      const float v = xData[i];
      const float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
      yData[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
#endif

    for (uint32_t i = 0; i < N; ++i) {
      yData[i] = static_cast<float>(tmp_out[i]);
    }
    return GraphStatus::Success;
  } else if (in_info.dtype == DType::QUInt8) {
    const uint8_t* xData = static_cast<const uint8_t*>(x.raw_data_const());
    uint8_t* yData = static_cast<uint8_t*>(y.raw_data());

    const float x_scale = in_info.scale;
    const float y_scale = out_info.scale;
    const int32_t x_zero = in_info.offset;
    const int32_t y_zero = out_info.offset;

    alignas(128) static uint8_t lut[256];
    static bool lut_init = false;
    if (!lut_init) {
      for (int i = 0; i < 256; ++i) {
        float x_f = (i - x_zero) * x_scale;
        float inner = 0.7978845608f * (x_f + 0.044715f * x_f * x_f * x_f);
        float y_f = 0.5f * x_f * (1.0f + std::tanh(inner));
        int y_q = static_cast<int>(std::round(y_f / y_scale)) + y_zero;
        lut[i] = static_cast<uint8_t>(std::clamp(y_q, 0, 255));
      }
      lut_init = true;
    }
    for (uint32_t i = 0; i < N; ++i) {
      yData[i] = lut[xData[i]];
    }
    return GraphStatus::Success;
  } else {
    return GraphStatus::ErrorFatal;
  }
}

__attribute__((unused)) static float fastgeluCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0; // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_FastGelu);
