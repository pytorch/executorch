
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSGraphTensor*
binaryOpTensor(
  MPSGraphTensor* primaryTensor,
  MPSGraphTensor* secondaryTensor,
  MPSGraph* mpsGraph,
  std::function<MPSGraphTensor*(MPSGraphTensor*, MPSGraphTensor*)> binaryOpFunction) {
  MPSDataType mpsInputDataType = [primaryTensor dataType];
  MPSDataType mpsOtherDataType = [secondaryTensor dataType];

  exec_aten::ScalarType inputDataType = getScalarType(mpsInputDataType);
  exec_aten::ScalarType otherDataType = getScalarType(mpsOtherDataType);

  MPSGraphTensor* primaryCastTensor = primaryTensor;
  MPSGraphTensor* secondaryCastTensor = secondaryTensor;
  exec_aten::ScalarType commonDataType = promoteTypes(inputDataType, otherDataType);
  if (inputDataType != commonDataType) {
    primaryCastTensor = castMPSTensor(mpsGraph, primaryTensor, commonDataType);
  }
  if (otherDataType != commonDataType) {
    secondaryCastTensor = castMPSTensor(mpsGraph, secondaryTensor, commonDataType);
  }

  return binaryOpFunction(primaryCastTensor, secondaryCastTensor);
}

/*
Helper macro to create an MPSGraph node based on the serialized data from the FlatBuffer.
It takes 2 inputs, an alpha parameter and returns one output. Couple operators from PyTorch,
such as torch.sub, torch.add take an additional alpha param.
More info at https://pytorch.org/docs/stable/generated/torch.sub.html.
*/
#define REGISTER_BINARY_WITH_ALPHA_OP(aot_name, graph_op)                                       \
Error                                                                                           \
MPSGraphBuilder::mps##aot_name##Op(NodePtr nodePtr)  {                                          \
auto graphNode = nodePtr->mpsnode_union_as_MPS##aot_name();                                     \
  ET_LOG(                                                                                       \
    Debug, "%s: (%d, %d) -> %d",                                                                \
    __FUNCTION__,                                                                               \
    graphNode->input1_id(),                                                                     \
    graphNode->input2_id(),                                                                     \
    graphNode->output_id()                                                                      \
  );                                                                                            \
                                                                                                \
  _idToMPSGraphTensor[graphNode->output_id()] = binaryOpTensor(                                 \
    getMPSGraphTensor(graphNode->input1_id()),                                                  \
    getMPSGraphTensor(graphNode->input2_id()),                                                  \
    _mpsGraph,                                                                                  \
    [&](MPSGraphTensor* primaryCastTensor,                                                      \
        MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {                               \
      if (graphNode->alpha() != 1.0) {                                                          \
        MPSGraphTensor* alphaTensor = [_mpsGraph constantWithScalar:graphNode->alpha()          \
                                                             shape:@[@1]                        \
                                                          dataType:primaryCastTensor.dataType]; \
        secondaryCastTensor = [_mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor    \
                                                        secondaryTensor:alphaTensor             \
                                                                   name:nil];                   \
      }                                                                                         \
      return [_mpsGraph graph_op##WithPrimaryTensor:primaryCastTensor                           \
                                  secondaryTensor:secondaryCastTensor                           \
                                             name:nil];                                         \
    }                                                                                           \
  );                                                                                            \
  return Error::Ok;                                                                             \
}

/*
Helper macro to create an MPSGraph node based on the serialized data from the FlatBuffer.
It takes 2 inputs and returns one output.
*/
#define REGISTER_BINARY_OP(aot_name, graph_op)                                  \
Error                                                                           \
MPSGraphBuilder::mps##aot_name##Op(NodePtr nodePtr)  {                          \
auto graphNode = nodePtr->mpsnode_union_as_MPS##aot_name();                     \
  ET_LOG(                                                                       \
    Debug, "%s: (%d, %d) -> %d",                                                \
    __FUNCTION__,                                                               \
    graphNode->input1_id(),                                                     \
    graphNode->input2_id(),                                                     \
    graphNode->output_id()                                                      \
  );                                                                            \
                                                                                \
  _idToMPSGraphTensor[graphNode->output_id()] = binaryOpTensor(                 \
    getMPSGraphTensor(graphNode->input1_id()),                                  \
    getMPSGraphTensor(graphNode->input2_id()),                                  \
    _mpsGraph,                                                                  \
    [&](MPSGraphTensor* primaryCastTensor,                                      \
        MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {               \
      return [_mpsGraph graph_op##WithPrimaryTensor:primaryCastTensor           \
                                  secondaryTensor:secondaryCastTensor           \
                                             name:nil];                         \
    }                                                                           \
  );                                                                            \
                                                                                \
  return Error::Ok;                                                             \
}

#define REGISTER_BITWISE_BINARY_OP(aot_name, graph_op)                             \
Error                                                                              \
MPSGraphBuilder::mps##aot_name##Op(NodePtr nodePtr)  {                             \
auto graphNode = nodePtr->mpsnode_union_as_MPS##aot_name();                        \
  ET_LOG(                                                                          \
    Debug, "%s: (%d, %d) -> %d",                                                   \
    __FUNCTION__,                                                                  \
    graphNode->input1_id(),                                                        \
    graphNode->input2_id(),                                                        \
    graphNode->output_id()                                                         \
  );                                                                               \
  ET_CHECK_OR_RETURN_ERROR(                                                        \
    is_macos_13_or_newer(), NotSupported,                                              \
    "%s supported by MPS on MacOS13.0+/iOS16.1+", #aot_name);                       \
                                                                                   \
  _idToMPSGraphTensor[graphNode->output_id()] = binaryOpTensor(                    \
    getMPSGraphTensor(graphNode->input1_id()),                                     \
    getMPSGraphTensor(graphNode->input2_id()),                                     \
    _mpsGraph,                                                                     \
    [&](MPSGraphTensor* primaryCastTensor,                                         \
        MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {                  \
      MPSDataType mpsInputDataType = [primaryCastTensor dataType];                 \
      if (getScalarType(mpsInputDataType) == ScalarType::Bool) {                   \
        return [_mpsGraph logical##graph_op##WithPrimaryTensor:primaryCastTensor   \
                                             secondaryTensor:secondaryCastTensor   \
                                                        name:nil];                 \
      }                                                                            \
      return [_mpsGraph bitwise##graph_op##WithPrimaryTensor:primaryCastTensor     \
                                             secondaryTensor:secondaryCastTensor   \
                                                        name:nil];                 \
    }                                                                              \
  );                                                                               \
                                                                                   \
  return Error::Ok;                                                                \
}

// Arithmetic Binary Ops
REGISTER_BINARY_WITH_ALPHA_OP(Add, addition)
REGISTER_BINARY_WITH_ALPHA_OP(Sub, subtraction)
REGISTER_BINARY_OP(Mul, multiplication)
REGISTER_BINARY_OP(Pow, power)
REGISTER_BINARY_OP(Minimum, minimum)

// Boolean Binary ops
REGISTER_BINARY_OP(Eq, equal)
REGISTER_BINARY_OP(Ne, notEqual)
REGISTER_BINARY_OP(Ge, greaterThanOrEqualTo)
REGISTER_BINARY_OP(Gt, greaterThan)
REGISTER_BINARY_OP(Le, lessThanOrEqualTo)
REGISTER_BINARY_OP(Lt, lessThan)

// Bitwise Binary ops
REGISTER_BITWISE_BINARY_OP(BitwiseAnd, AND)
REGISTER_BITWISE_BINARY_OP(BitwiseOr, OR)
REGISTER_BITWISE_BINARY_OP(BitwiseXor, XOR)

#undef REGISTER_BINARY_WITH_ALPHA_OP
#undef REGISTER_BINARY_OP

static
MPSGraphTensor* mpsTruncTensor(MPSGraphTensor* inputTensor, MPSGraph* mpsGraph) {
  // Rounding is a no-op for integral types, and also a reasonable workaround
  // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
  // See https://github.com/pytorch/pytorch/issues/84995
  bool isFloatInput = ([inputTensor dataType] & MPSDataTypeFloatBit) != 0;
  if (!isFloatInput) {
    return inputTensor;
  }

  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS)) {
    MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                          secondaryTensor:zeroTensor
                                                                     name:nil];
    return [mpsGraph selectWithPredicateTensor:predicateTensor
                           truePredicateTensor:[mpsGraph ceilWithTensor:inputTensor name:nil]
                          falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                          name:nil];
  } else {
    return [mpsGraph truncateWithTensor:inputTensor
                                    name:nil];
  }
};

static
MPSGraphTensor* divModeTemplate(
  MPSGraphTensor* primaryTensor,
  MPSGraphTensor* secondaryTensor,
  std::optional<flatbuffers::string_view> rounding_mode,
  MPSGraph* mpsGraph,
  const std::string& op_name) {
  MPSDataType mpsInputDataType = [primaryTensor dataType];
  ScalarType inputDataType = getScalarType(mpsInputDataType);

  if(rounding_mode.has_value() && *rounding_mode == "trunc"){
    ET_CHECK_MSG(inputDataType != ScalarType::Half,
                "MPS: does not support trunc_divide op with float16 input");
  }

  auto divOpFunc = [&](MPSGraphTensor* primaryCastTensor,
                      MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {
    bool isFloatInput = ([primaryCastTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!isFloatInput && rounding_mode.has_value() && (*rounding_mode == "floor" || *rounding_mode == "trunc")) {
      primaryCastTensor = [mpsGraph castTensor:primaryCastTensor
                                        toType:MPSDataTypeFloat32
                                          name:@"primaryCastTensor"];
      secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                          toType:MPSDataTypeFloat32
                                            name:@"secondaryCastTensor"];
    }
    MPSGraphTensor* divTensor =  [mpsGraph divisionWithPrimaryTensor:primaryCastTensor
                                                      secondaryTensor:secondaryCastTensor
                                                                name:nil];

    // Rounding is a no-op for integral types, and also a reasonable workaround
    // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
    // See https://github.com/pytorch/pytorch/issues/84995
    bool isFloatOutput = ([divTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!rounding_mode.has_value() || !isFloatOutput) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      auto truncTensor =  mpsTruncTensor(divTensor, mpsGraph);
      if (op_name == "Fmod") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:truncTensor
                                                    secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                  name:nil];
      }
      return truncTensor;
    } else if (*rounding_mode == "floor") {
      MPSGraphTensor* floorTensor = [mpsGraph floorWithTensor:divTensor name:nil];
      if (op_name == "Remainder") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:floorTensor
                                                    secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                  name:nil];
      }
      return floorTensor;
    } else {
      assert(0 && "Invalid rounding mode\n");
    }
    return nullptr;
  };
  return binaryOpTensor(primaryTensor, secondaryTensor, mpsGraph, divOpFunc);
}

#define REGISTER_DIV_OP(aot_name, round_mode)                                         \
Error                                                                                 \
MPSGraphBuilder::mps##aot_name##Op(NodePtr nodePtr)  {                                \
  auto graphNode = nodePtr->mpsnode_union_as_MPS##aot_name();                         \
  ET_LOG(                                                                             \
    Debug, "%s: (%d, %d) -> %d",                                                      \
    __FUNCTION__,                                                                     \
    graphNode->input1_id(),                                                           \
    graphNode->input2_id(),                                                           \
    graphNode->output_id()                                                            \
  );                                                                                  \
                                                                                      \
  auto strView = graphNode->rounding_mode() != nullptr ?                              \
    std::make_optional(graphNode->rounding_mode()->string_view()) : round_mode;       \
                                                                                      \
  _idToMPSGraphTensor[graphNode->output_id()] = divModeTemplate(                      \
    getMPSGraphTensor(graphNode->input1_id()),                                        \
    getMPSGraphTensor(graphNode->input2_id()),                                        \
    strView,                                                                          \
    _mpsGraph,                                                                        \
    #aot_name                                                                         \
  );                                                                                  \
                                                                                      \
  return Error::Ok;                                                                   \
}

REGISTER_DIV_OP(Div, std::nullopt)
REGISTER_DIV_OP(Fmod, "trunc")
REGISTER_DIV_OP(Remainder, "floor")

#undef REGISTER_DIV_OP


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
