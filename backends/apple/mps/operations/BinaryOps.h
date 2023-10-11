//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//
// clang-format off
#pragma once

#define REGISTER_PYBIND11_MPS_BINARY_OP(py11_export_name, graph_op)                                    \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input, PyMPSGraphTensor* other) {    \
return self.binaryOpTensor(                                                                            \
  static_cast<MPSGraphTensor*>(input), static_cast<MPSGraphTensor*>(other), py11_export_name,          \
  [&](MPSGraphTensor* primaryCastTensor, MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {     \
    return [self.getMPSGraph() graph_op##WithPrimaryTensor:primaryCastTensor                           \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
  });                                                                                                  \
})                                                                                                     \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input, float sc) {                   \
return self.binaryOpWithScalar(                                                                        \
  static_cast<MPSGraphTensor*>(input), sc, py11_export_name,                                           \
  [&](MPSGraphTensor* primaryCastTensor, MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {     \
    MPSDataType mpsInputDataType = [primaryCastTensor dataType];                                       \
    return [self.getMPSGraph() graph_op##WithPrimaryTensor:primaryCastTensor                           \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
  });                                                                                                  \
})                                                                                                     \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input, int sc) {                     \
return self.binaryOpWithScalar(                                                                        \
  static_cast<MPSGraphTensor*>(input), sc, py11_export_name,                                           \
  [&](MPSGraphTensor* primaryCastTensor, MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {     \
    MPSDataType mpsInputDataType = [primaryCastTensor dataType];                                       \
    return [self.getMPSGraph() graph_op##WithPrimaryTensor:primaryCastTensor                           \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
  });                                                                                                  \
})                                                                                                     \

#define REGISTER_PYBIND11_MPS_BITWISE_BINARY_OP(py11_export_name, graph_op)                            \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input, PyMPSGraphTensor* other) {    \
return self.binaryOpTensor(                                                                            \
  static_cast<MPSGraphTensor*>(input), static_cast<MPSGraphTensor*>(other), py11_export_name,          \
  [&](MPSGraphTensor* primaryCastTensor, MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {     \
    MPSDataType mpsInputDataType = [primaryCastTensor dataType];                                       \
    if (getScalarType(mpsInputDataType) == ScalarType::Bool) {                                         \
      return [self.getMPSGraph() logical##graph_op##WithPrimaryTensor:primaryCastTensor                \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
    }                                                                                                  \
    return [self.getMPSGraph() bitwise##graph_op##WithPrimaryTensor:primaryCastTensor                  \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
  });                                                                                                  \
})                                                                                                     \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input, int sc) {                     \
return self.binaryOpWithScalar(                                                                        \
  static_cast<MPSGraphTensor*>(input), sc, py11_export_name,                                           \
  [&](MPSGraphTensor* primaryCastTensor, MPSGraphTensor* secondaryCastTensor) -> MPSGraphTensor* {     \
    MPSDataType mpsInputDataType = [primaryCastTensor dataType];                                       \
    if (getScalarType(mpsInputDataType) == ScalarType::Bool) {                                         \
      return [self.getMPSGraph() logical##graph_op##WithPrimaryTensor:primaryCastTensor                \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
    }                                                                                                  \
    return [self.getMPSGraph() bitwise##graph_op##WithPrimaryTensor:primaryCastTensor                  \
                                           secondaryTensor:secondaryCastTensor                         \
                                                      name:nil];                                       \
  });                                                                                                  \
})                                                                                                     \
// clang-format on
