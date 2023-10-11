//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"
#include "UnaryOps.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::unaryOpTensor(
  MPSGraphTensor* inputTensor,
  const std::string& op_name,
  std::function<MPSGraphTensor*(MPSGraphTensor*)> unaryOpFunction) {
    return unaryOpFunction(inputTensor);
  }

PyMPSGraphTensor*
MPSGraphModule::cumsum(
  MPSGraphTensor* inputTensor,
  int dim
) {
  return [mpsGraph cumulativeSumWithTensor:inputTensor
                                      axis:dim
                                      name:@"cumsum"];
}

}//namespace
