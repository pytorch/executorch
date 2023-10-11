//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*>
MPSGraphModule::minDim(MPSGraphTensor* inputTensor,
                     int dim,
                     bool keep_dims) {

  const int input_dims = inputTensor.shape.count;
  int wrapped_dim = torch::maybe_wrap_dim(dim, input_dims);

  MPSGraphTensor* minTensor =  [mpsGraph reductionMinimumWithTensor:inputTensor
                                                               axis:wrapped_dim
                                                               name:@"minTensor"];

  MPSGraphTensor* indicesTensor = [mpsGraph reductionArgMinimumWithTensor:inputTensor
                                              axis:wrapped_dim
                                              name:@"argminTensor"];

  if ([indicesTensor dataType] != MPSDataTypeInt64) {
    indicesTensor = [mpsGraph castTensor:indicesTensor
                                  toType:MPSDataTypeInt64
                                    name:@"argminTensor/cast"];
  }

  if(!keep_dims) {
    minTensor = [mpsGraph squeezeTensor:minTensor
                                    axis:wrapped_dim
                                    name:@"minTensor/squeezed"];
    indicesTensor = [mpsGraph squeezeTensor:indicesTensor
                                    axis:wrapped_dim
                                    name:@"argminTensor/squeezed"];
  }

  return std::make_tuple(minTensor, indicesTensor);
}

std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*>
MPSGraphModule::maxDim(MPSGraphTensor* inputTensor,
                     int dim,
                     bool keep_dims) {

  const int input_dims = inputTensor.shape.count;
  int wrapped_dim = torch::maybe_wrap_dim(dim, input_dims);

  MPSGraphTensor* maxTensor =  [mpsGraph reductionMaximumWithTensor:inputTensor
                                                               axis:wrapped_dim
                                                               name:@"maxTensor"];

  MPSGraphTensor* indicesTensor = [mpsGraph reductionArgMaximumWithTensor:inputTensor
                                              axis:wrapped_dim
                                              name:@"argmaxTensor"];

  if ([indicesTensor dataType] != MPSDataTypeInt64) {
    indicesTensor = [mpsGraph castTensor:indicesTensor
                                  toType:MPSDataTypeInt64
                                    name:@"argmaxTensor/cast"];
  }

  if(!keep_dims) {
    maxTensor = [mpsGraph squeezeTensor:maxTensor
                                    axis:wrapped_dim
                                    name:@"minTensor/squeezed"];
    indicesTensor = [mpsGraph squeezeTensor:indicesTensor
                                    axis:wrapped_dim
                                    name:@"argminTensor/squeezed"];
  }

  return std::make_tuple(maxTensor, indicesTensor);
}

PyMPSGraphTensor*
MPSGraphModule::amax(MPSGraphTensor* inputTensor,
                     IntArrayRef dims,
                     bool keep_dims) {

  const int input_dims = inputTensor.shape.count;
  NSMutableArray<NSNumber*>* dimArray = [NSMutableArray array];
  for(int dim: dims) {
    int wrapped_dim = torch::maybe_wrap_dim(dim, input_dims);
    [dimArray addObject:[NSNumber numberWithInt:wrapped_dim]];
  }

  MPSGraphTensor* amaxTensor = [mpsGraph reductionMaximumWithTensor:inputTensor
                                                               axes:dimArray
                                                               name:@"AmaxTensor"];
  if(!keep_dims) {
    amaxTensor = [mpsGraph squeezeTensor:amaxTensor
                                    axes:dimArray
                                    name:@"AmaxTensor/squeezed"];
  }

  return amaxTensor;
}

PyMPSGraphTensor*
MPSGraphModule::amin(MPSGraphTensor* inputTensor,
                     IntArrayRef dims,
                     bool keep_dims) {

  const int input_dims = inputTensor.shape.count;
  NSMutableArray<NSNumber*>* dimArray = [NSMutableArray array];
  for(int dim: dims) {
    int wrapped_dim = torch::maybe_wrap_dim(dim, input_dims);
    [dimArray addObject:[NSNumber numberWithInt:wrapped_dim]];
  }

  MPSGraphTensor* aminTensor = [mpsGraph reductionMinimumWithTensor:inputTensor
                                                               axes:dimArray
                                                               name:@"AminTensor"];
  if(!keep_dims) {
    aminTensor = [mpsGraph squeezeTensor:aminTensor
                                    axes:dimArray
                                    name:@"AminTensor/squeezed"];
  }

  return aminTensor;
}

PyMPSGraphTensor*
MPSGraphModule::argmax(MPSGraphTensor* inputTensor,
                     int64_t dim,
                     bool keep_dims,
                     bool flatten) {

  auto dim_ = maybe_wrap_dim(dim, inputTensor.shape.count);

  MPSGraphTensor* output = inputTensor;
  //In case the dimension is not specified, expectation is to return index
  //of entry in flattened input tensor
  if(flatten) {
    NSInteger nElems = 0;
    for (NSNumber *num in inputTensor.shape) {
      nElems *= [num intValue];
    }
    dim_ = 0;
    output =  [mpsGraph reshapeTensor:inputTensor
                            withShape:@[@-1]
                            name:nil];
  }

  output = [mpsGraph reductionArgMaximumWithTensor:output
                                              axis:dim_
                                              name:@"ArgmaxTensor"];
  if(!keep_dims && !flatten) {
    output = [mpsGraph squeezeTensor:output
                                axis:dim_
                                name:@"ArgmaxTensor/squeezed"];
  }

  if ([output dataType] != MPSDataTypeInt64) {
    output = [mpsGraph castTensor:output
                        toType:MPSDataTypeInt64
                        name:@"ArgmaxTensor/cast"];
  }

  return output;
}

PyMPSGraphTensor*
MPSGraphModule::argmin(MPSGraphTensor* inputTensor,
                     int64_t dim,
                     bool keep_dims,
                     bool flatten) {

  auto dim_ = maybe_wrap_dim(dim, inputTensor.shape.count);

  MPSGraphTensor* output = inputTensor;
  //In case the dimension is not specified, expectation is to return index
  //of entry in flattened input tensor
  if(flatten) {
    NSInteger nElems = 0;
    for (NSNumber *num in inputTensor.shape) {
      nElems *= [num intValue];
    }
    dim_ = 0;
    output =  [mpsGraph reshapeTensor:inputTensor
                            withShape:@[@-1]
                            name:nil];
  }

  output = [mpsGraph reductionArgMinimumWithTensor:output
                                              axis:dim_
                                              name:@"ArgminTensor"];
  if(!keep_dims && !flatten) {
    output = [mpsGraph squeezeTensor:output
                                axis:dim_
                                name:@"ArgminTensor/squeezed"];
  }

  if ([output dataType] != MPSDataTypeInt64) {
    output = [mpsGraph castTensor:output
                        toType:MPSDataTypeInt64
                        name:@"ArgminTensor/cast"];
  }

  return output;
}

PyMPSGraphTensor*
MPSGraphModule::mean(MPSGraphTensor* inputTensor,
                     IntArrayRef dims,
                     bool keep_dims) {

  //MPSGraph wants negative axes to be converted to positive
  const int input_dims = [inputTensor.shape count];
  NSMutableArray<NSNumber*>* dimArray = [NSMutableArray array];
  for(int i = 0; i<dims.size(); i++) {
    int dim = dims[i];
    if(dim<0) {
      dim = input_dims + dim;
    }
    [dimArray addObject:[NSNumber numberWithInt:dim]];
  }
  //Reverting back to get the ordering back to slowest axis first as MPSGraph expects
  dimArray = [[[dimArray reverseObjectEnumerator] allObjects] mutableCopy];

  MPSGraphTensor* meanTensor = [mpsGraph meanOfTensor:inputTensor
                          axes:dimArray
                          name:@"Mean"];
  if(!keep_dims) {
    meanTensor = [mpsGraph squeezeTensor:meanTensor
                                    axes:dimArray
                                    name:@"Mean/squeezed"];
  }

  return meanTensor;

}
}//namespace mps

