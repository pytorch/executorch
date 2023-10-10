//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//
// clang-format off
#pragma once

#define REGISTER_PYBIND11_MPS_UNARY_OP(py11_export_name, graph_op)                                    \
.def(py11_export_name, [](MPSGraphModule& self, PyMPSGraphTensor* input) {                            \
return self.unaryOpTensor(                                                                            \
    static_cast<MPSGraphTensor*>(input), py11_export_name,                                            \
    [&](MPSGraphTensor* inputTensor) -> MPSGraphTensor* {                                             \
      return [self.getMPSGraph() graph_op##WithTensor:inputTensor                                     \
                                                        name:nil];                                    \
    });                                                                                               \
})
// clang-format on
