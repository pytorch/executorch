//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::index_select(MPSGraphTensor* inputTensor, int64_t dim, MPSGraphTensor* indexTensor) {
  dim = maybe_wrap_dim(dim, inputTensor.shape.count);

  MPSGraphTensor* castIndexTensor = indexTensor;
  if(castIndexTensor.dataType != MPSDataTypeInt32) {
    castIndexTensor = [mpsGraph castTensor:indexTensor
                                    toType:MPSDataTypeInt32
                                      name:nil];
  }

  MPSGraphTensor* outputTensor = [mpsGraph gatherWithUpdatesTensor: inputTensor
                                                     indicesTensor: castIndexTensor
                                                              axis: dim
                                                   batchDimensions: 0
                                                              name: nil];
  return outputTensor;
}

} // namespace at::native
