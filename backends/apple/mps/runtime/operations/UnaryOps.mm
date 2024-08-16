
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
unaryOpTensor(
  MPSGraphTensor* inputTensor,
  MPSGraph* mpsGraph,
  std::function<MPSGraphTensor*(MPSGraphTensor*)> unaryOpFunction) {
    return unaryOpFunction(inputTensor);
}

Error
MPSGraphBuilder::mpsBitwiseNotOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSBitwiseNot();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSDataType mpsInputDataType = [inputTensor dataType];
  if (getScalarType(mpsInputDataType) == ScalarType::Bool) {
    _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph notWithTensor:inputTensor name:nil];
  } else {
    ET_CHECK_OR_RETURN_ERROR(
      is_macos_13_or_newer(), NotSupported,
      "mpsBitwiseNotOp supported by MPS on MacOS13.0+/iOS16.1+");
    _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph bitwiseNOTWithTensor:inputTensor name:nil];
  }

  return Error::Ok;
}

#define REGISTER_UNARY_OP(aot_name, graph_op)                                                   \
Error                                                                                           \
MPSGraphBuilder::mps##aot_name##Op(NodePtr nodePtr)  {                                          \
  auto graphNode = static_cast<const mpsgraph::_MPSNode1x1 *>(nodePtr->mpsnode_union());        \
  ET_LOG(                                                                                       \
    Debug, "%s: %d -> %d",                                                                      \
    __FUNCTION__,                                                                               \
    graphNode->input1_id(),                                                                     \
    graphNode->output_id()                                                                      \
  );                                                                                            \
  _idToMPSGraphTensor[graphNode->output_id()] = unaryOpTensor(                                  \
    getMPSGraphTensor(graphNode->input1_id()),                                                  \
    _mpsGraph,                                                                                  \
    [&](MPSGraphTensor* inputTensor) -> MPSGraphTensor* {                                       \
      return [_mpsGraph graph_op##WithTensor:inputTensor                                        \
                                        name:nil];                                              \
    }                                                                                           \
  );                                                                                            \
  return Error::Ok;                                                                             \
}

REGISTER_UNARY_OP(Exp, exponent)
REGISTER_UNARY_OP(Exp2, exponentBase2)
REGISTER_UNARY_OP(Reciprocal, reciprocal)
REGISTER_UNARY_OP(Sqrt, squareRoot)
REGISTER_UNARY_OP(Neg, negative)
REGISTER_UNARY_OP(Log, logarithm)
REGISTER_UNARY_OP(Log10, logarithmBase10)
REGISTER_UNARY_OP(Log2, logarithmBase2)
REGISTER_UNARY_OP(Erf, erf)
REGISTER_UNARY_OP(Floor, floor)
REGISTER_UNARY_OP(Ceil, ceil)
REGISTER_UNARY_OP(Rsqrt, reverseSquareRoot)
REGISTER_UNARY_OP(Sigmoid, sigmoid)
REGISTER_UNARY_OP(Sin, sin)
REGISTER_UNARY_OP(Sign, sign)
REGISTER_UNARY_OP(Cos, cos)
REGISTER_UNARY_OP(Tan, tan)
REGISTER_UNARY_OP(Abs,  absolute)
REGISTER_UNARY_OP(Asin, asin)
REGISTER_UNARY_OP(Acos, acos)
REGISTER_UNARY_OP(Atan, atan)
REGISTER_UNARY_OP(Sinh, sinh)
REGISTER_UNARY_OP(Cosh, cosh)
REGISTER_UNARY_OP(Tanh, tanh)
REGISTER_UNARY_OP(Asinh, asinh)
REGISTER_UNARY_OP(Acosh, acosh)
REGISTER_UNARY_OP(Atanh, atanh)
REGISTER_UNARY_OP(Isnan, isNaN)
REGISTER_UNARY_OP(Isinf, isInfinite)
REGISTER_UNARY_OP(Round, round)
REGISTER_UNARY_OP(LogicalNot, not)


Error
MPSGraphBuilder::mpsNormCdfOp(NodePtr nodePtr)  {
  auto graphNode = static_cast<const mpsgraph::_MPSNode1x1 *>(nodePtr->mpsnode_union());
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output_id()
  );
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  auto dataType = [inputTensor dataType];
  const float SQRT1_2 = 0.707106781186547524400844362104849039f;
  MPSGraphTensor *sqrt1_2 = [_mpsGraph constantWithScalar:SQRT1_2
                                                    shape:@[@1]
                                                 dataType:dataType];
  MPSGraphTensor *onef = [_mpsGraph constantWithScalar:1.0f
                                                shape:@[@1]
                                             dataType:dataType];
  MPSGraphTensor *halff = [_mpsGraph constantWithScalar:0.5f
                                                 shape:@[@1]
                                              dataType:dataType];

  MPSGraphTensor *erfTensor = [_mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                         secondaryTensor:sqrt1_2
                                                                    name:nil];
  erfTensor = [_mpsGraph erfWithTensor:erfTensor name:nil];
  erfTensor = [_mpsGraph additionWithPrimaryTensor:erfTensor
                                   secondaryTensor:onef
                                              name:nil];
  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph multiplicationWithPrimaryTensor:erfTensor
                               secondaryTensor:halff
                                          name:nil];

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
