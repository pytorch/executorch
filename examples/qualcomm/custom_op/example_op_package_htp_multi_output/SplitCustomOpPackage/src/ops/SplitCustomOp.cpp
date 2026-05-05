//==============================================================================
// Auto Generated Code for SplitCustomOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_SplitCustomOp);

// op execute function declarations
template <typename TensorType>
GraphStatus splitcustomopImpl(
    TensorType& first_half,
    TensorType& second_half,
    const TensorType& in_0);

// forward declaration of sample cost function
static float splitcustomopCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default
 * flag (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((splitcustomopImpl<Tensor>), "SplitCustomOp")
 */
DEF_PACKAGE_OP((splitcustomopImpl<Tensor>), "SplitCustomOp")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or more flags,
 * FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX, RESOURCE_HMX(not
 * supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((splitcustomopImpl<PlainFloatTensor>),
 * "SplitCustomOp", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g.
 * DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((splitcustomopImpl<PlainFloatTensor>),
 * "SplitCustomOp", splitcustomopCostFunc, Flags::RESOURCE_HVX)
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

template <typename TensorType>
GraphStatus splitcustomopImpl(
    TensorType& first_half,
    TensorType& second_half,
    const TensorType& in_0)

{
  /*
   * add code here
   * */
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */
  DTypeScaleOff input_intfc = in_0.get_dtype_intfc();

  if (input_intfc.dtype != DType::Float32 &&
      input_intfc.dtype != DType::QUInt8) {
    return GraphStatus::ErrorPrecision;
  }

  // Input shape: [N, H, W, C] (NHWC). Split along C (last dim).
  const size_t N = in_0.dim(0);
  const size_t H = in_0.dim(1);
  const size_t W = in_0.dim(2);
  const size_t C = in_0.dim(3);
  const size_t half = C / 2;

  if (input_intfc.dtype == DType::Float32) {
    const float* p_in = static_cast<const float*>(in_0.raw_data_const());
    float* p_first = static_cast<float*>(first_half.raw_data());
    float* p_second = static_cast<float*>(second_half.raw_data());

    for (size_t n = 0; n < N; ++n) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          const float* row = p_in + ((n * H + h) * W + w) * C;
          float* row_first = p_first + ((n * H + h) * W + w) * half;
          float* row_second = p_second + ((n * H + h) * W + w) * half;
          for (size_t c = 0; c < half; ++c) {
            row_first[c] = row[c];
            row_second[c] = row[c + half];
          }
        }
      }
    }
  } else { // QUInt8
    const uint8_t* p_in = static_cast<const uint8_t*>(in_0.raw_data_const());
    uint8_t* p_first = static_cast<uint8_t*>(first_half.raw_data());
    uint8_t* p_second = static_cast<uint8_t*>(second_half.raw_data());

    for (size_t n = 0; n < N; ++n) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          const uint8_t* row = p_in + ((n * H + h) * W + w) * C;
          uint8_t* row_first = p_first + ((n * H + h) * W + w) * half;
          uint8_t* row_second = p_second + ((n * H + h) * W + w) * half;
          for (size_t c = 0; c < half; ++c) {
            row_first[c] = row[c];
            row_second[c] = row[c + half];
          }
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float splitcustomopCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0; // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_SplitCustomOp);
