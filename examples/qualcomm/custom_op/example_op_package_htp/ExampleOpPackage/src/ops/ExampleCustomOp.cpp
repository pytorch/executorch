//==============================================================================
// Auto Generated Code for ExampleOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"
#ifdef __hexagon__
#include "HAP_farf.h"
#else /* __hexagon__ */
#include <cstdio>
#define FARF(level, fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#endif /* __hexagon__ */

BEGIN_PKG_OP_DEFINITION(PKG_ExampleCustomOp);

// op execute function declarations
template <typename TensorType>
GraphStatus examplecustomopImpl(TensorType& out_0, const TensorType& in_0);

// forward declaration of sample cost function
static float examplecustomopCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default
 * flag (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((examplecustomopImpl<Tensor>), "ExampleCustomOp")
 */
DEF_PACKAGE_OP((examplecustomopImpl<Tensor>), "ExampleCustomOp")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or more flags,
 * FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX, RESOURCE_HMX(not
 * supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((examplecustomopImpl<PlainFloatTensor>),
 * "ExampleCustomOp", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g.
 * DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((examplecustomopImpl<PlainFloatTensor>),
 * "ExampleCustomOp", examplecustomopCostFunc, Flags::RESOURCE_HVX)
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
GraphStatus examplecustomopImpl(TensorType& out_0, const TensorType& in_0)

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
  const size_t input_num_elements = in_0.total_storage_elements();
  DTypeScaleOff input_intfc = in_0.get_dtype_intfc();

  FARF(
      ALWAYS,
      "[QNN ExecuTorch Op Package test] "
      "input num_elem: %zu, dtype %d, scale %f, offset %d",
      input_num_elements,
      input_intfc.dtype,
      input_intfc.scale,
      input_intfc.offset);

  if (input_intfc.dtype != DType::Float32 &&
      input_intfc.dtype != DType::QUInt8) {
    FARF(
        ALWAYS,
        "[QNN ExecuTorch Op Package test]"
        "[Error] The datatype of input is %d, not float32(%d) nor uint8(%d)",
        input_intfc.dtype,
        DType::Float32,
        DType::QUInt8);
    return GraphStatus::ErrorPrecision;
  }

  const size_t output_num_elements = out_0.total_storage_elements();
  DTypeScaleOff out_intfc = out_0.get_dtype_intfc();
  FARF(
      ALWAYS,
      "[QNN ExecuTorch Op Package test] "
      "out num_elem: %zu, dtype %d, scale %f, offset %d",
      output_num_elements,
      out_intfc.dtype,
      out_intfc.scale,
      out_intfc.offset);
  if (out_intfc.dtype != DType::Float32 && out_intfc.dtype != DType::QUInt8) {
    FARF(
        ALWAYS,
        "[QNN ExecuTorch Op Package test]"
        "[Error] The datatype of output is %d, not float32(%d) nor uint8(%d)",
        out_intfc.dtype,
        DType::Float32,
        DType::QUInt8);
    return GraphStatus::ErrorPrecision;
  }

  if (input_num_elements != output_num_elements) {
    FARF(
        ALWAYS,
        "[QNN ExecuTorch Op Package test]"
        "[Error] The number of input and output doesn't match. "
        "input_num_elements: %zu, output_num_elements: %zu",
        input_num_elements,
        output_num_elements);
    return GraphStatus::ErrorDimensions;
  }
  if (input_intfc.dtype == DType::Float32) {
    const float* p_input = static_cast<const float*>(in_0.raw_data_const());
    float* p_output = static_cast<float*>(out_0.raw_data());
    const int multiplier = 3;
    for (size_t i = 0; i < input_num_elements; ++i) {
      p_output[i] = multiplier * p_input[i];

      FARF(
          ALWAYS,
          "[QNN ExecuTorch Op Package test]"
          "input0[%zu]=%f, multiplier=%d, output[%zu]=%f",
          i,
          p_input[i],
          multiplier,
          i,
          p_output[i]);
    }
  } else if (input_intfc.dtype == DType::QUInt8) {
    const uint8_t* p_input = static_cast<const uint8_t*>(in_0.raw_data_const());
    uint8_t* p_output = static_cast<uint8_t*>(out_0.raw_data());
    const int multiplier = 3 * input_intfc.scale / out_intfc.scale;
    for (size_t i = 0; i < input_num_elements; ++i) {
      p_output[i] = multiplier * p_input[i];

      FARF(
          ALWAYS,
          "[QNN ExecuTorch Op Package test]"
          "input0[%zu]=%f, multiplier=%d, output[%zu]=%f",
          i,
          p_input[i],
          multiplier,
          i,
          p_output[i]);
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float examplecustomopCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0; // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_ExampleCustomOp);
