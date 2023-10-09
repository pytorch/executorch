//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor *MPSGraphModule::bitwiseNotTensor(MPSGraphTensor *inputTensor,
                                                   const std::string &op_name) {
  MPSDataType mpsInputDataType = [inputTensor dataType];
  if (getScalarType(mpsInputDataType) == ScalarType::Bool) {
    return [getMPSGraph() notWithTensor:inputTensor name:nil];
  }
  return [getMPSGraph() bitwiseNOTWithTensor:inputTensor name:nil];
}
} // namespace mps
