//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::pixel_shuffle(MPSGraphTensor* inputTensor, int upscale_factor) {

  const int ndims = inputTensor.shape.count;
  TORCH_CHECK(ndims >= 3, "pixel_shuffle requires tensor with at least 3 dimensions.");
  if (upscale_factor == 1) {
    return inputTensor;
  }
  TORCH_CHECK(inputTensor.shape[ndims - 3].intValue % (upscale_factor * upscale_factor) == 0, 
                "pixel_shuffle channels must be divisible by upscale factor squared.");

  return [mpsGraph depthToSpace2DTensor:inputTensor
                              widthAxis:ndims - 1
                             heightAxis:ndims - 2
                              depthAxis:ndims - 3
                              blockSize:upscale_factor
                   usePixelShuffleOrder:true
                                   name:@"pixel_shuffle"];

}

std::vector<PyMPSGraphTensor*>
MPSGraphModule::split_size(MPSGraphTensor* inputTensor, IntArrayRef split_sizes, int dim) {

  TORCH_CHECK(dim >=0 && dim < inputTensor.shape.count, 
              "split_copy: dim ", dim, " out of range for input tensor with ", inputTensor.shape.count, " dimensions");

  std::vector<PyMPSGraphTensor*> splitResults;
  NSArray<MPSGraphTensor*>* mpsGraphResults;

  mpsGraphResults = [mpsGraph splitTensor:inputTensor
                               splitSizes:getMPSShape(split_sizes)
                                     axis:dim
                                     name:@"split_size"];
                                     
  for (MPSGraphTensor* splitTensor in mpsGraphResults) {
    splitResults.push_back(splitTensor);
  }
  return splitResults;
}

std::vector<PyMPSGraphTensor*>
MPSGraphModule::split(MPSGraphTensor* inputTensor, int split_size, int dim) {

  TORCH_CHECK(dim >=0 && dim < inputTensor.shape.count, 
              "split_copy: dim ", dim, " out of range for input tensor with ", inputTensor.shape.count, " dimensions");
  TORCH_CHECK(split_size > 0 && split_size <= inputTensor.shape[dim].intValue,
              "split_copy: split_size ", split_size, " invalid for inputTensor dimension ", dim, " with length ", inputTensor.shape[dim].intValue);

  NSMutableArray* splits = [NSMutableArray array];
  NSNumber* splitSize = [NSNumber numberWithInt:split_size];
  int i = 1;

  while(split_size * i < inputTensor.shape[dim].intValue) {
    [splits addObject:splitSize];
    i++;
  }

  int splits_adjust = inputTensor.shape[dim].intValue - (split_size * i);
  if (splits_adjust < 0) {
    splits[i - 1] = [NSNumber numberWithInt:(split_size + splits_adjust)];
  }

  std::vector<PyMPSGraphTensor*> splitResults;
  NSArray<MPSGraphTensor*>* mpsGraphResults;

  mpsGraphResults = [mpsGraph splitTensor:inputTensor
                               splitSizes:splits
                                     axis:dim
                                     name:@"split"];
  
  for (MPSGraphTensor* splitTensor in mpsGraphResults) {
    splitResults.push_back(splitTensor);
  }
  return splitResults;
}

std::vector<PyMPSGraphTensor*>
MPSGraphModule::unbind(MPSGraphTensor* inputTensor, int dim) {

  std::vector<PyMPSGraphTensor*> unbindResults;

  for (int i = 0; i < inputTensor.shape[dim].intValue; i++) {
    unbindResults.push_back(
      [mpsGraph sliceTensor:inputTensor
                  dimension:dim
                      start:i
                     length:1
                       name:@"unbind"]
    );
  }
  return unbindResults;
}

PyMPSGraphTensor*
MPSGraphModule::slice(MPSGraphTensor* inputTensor,
                        int64_t dim,
                        c10::optional<int64_t> start,
                        c10::optional<int64_t> end,
                        int64_t step) {

  int64_t dim_len = inputTensor.shape[dim].intValue;
  // Unwrap optional values
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : dim_len;
  // Convert python style indices to compatible values
  start_val = start_val < 0 ? start_val + dim_len : start_val;
  end_val = end_val < 0 ? end_val + dim_len : end_val;
  start_val = start_val < 0 ? 0 : start_val;
  end_val = end_val < 0 ? 0 : end_val;
  start_val = start_val > dim_len ? dim_len : start_val;
  end_val = end_val > dim_len ? dim_len : end_val;

  // Define input arrays as required by MPSGraph api
  NSMutableArray<NSNumber*>* start_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  NSMutableArray<NSNumber*>* end_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  NSMutableArray<NSNumber*>* step_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  // Step needs to be set to one for all other dims
  for (int i = 0; i < inputTensor.shape.count; i++) {
    step_arr[i] = @1;
    end_arr[i] = inputTensor.shape[i];
    start_arr[i] = @0;
  }

  start_arr[dim] = [NSNumber numberWithInteger:start_val];
  end_arr[dim] = [NSNumber numberWithInteger:end_val];
  step_arr[dim] = [NSNumber numberWithInteger:step];

  return [mpsGraph sliceTensor:inputTensor
                        starts:start_arr
                          ends:end_arr
                       strides:step_arr
                          name:@"strided_slice"];
}

PyMPSGraphTensor*
MPSGraphModule::cat(int dim, py::args catTensors) {
  NSMutableArray<MPSGraphTensor*>* inputTensors = [NSMutableArray array];
  for (const auto i: c10::irange(catTensors.size())) {
    MPSGraphTensor* catTensor = static_cast<MPSGraphTensor*>(pybind11::cast<void*>(catTensors[i]));
    if (catTensor != nil)
      [inputTensors addObject:static_cast<MPSGraphTensor*>(pybind11::cast<void*>(catTensors[i]))];
  }
  return [mpsGraph concatTensors:inputTensors
                       dimension:dim
                            name:@"cat"];
}

PyMPSGraphTensor*
MPSGraphModule::stack(int dim, py::args stackTensors) {
  NSMutableArray<MPSGraphTensor*>* inputTensors = [NSMutableArray array];
  for (const auto i: c10::irange(stackTensors.size())) {
    [inputTensors addObject:static_cast<MPSGraphTensor*>(pybind11::cast<void*>(stackTensors[i]))];
  }
  return [mpsGraph stackTensors:inputTensors
                           axis:dim
                           name:@"stack"];
}

PyMPSGraphTensor*
MPSGraphModule::expand(MPSGraphTensor* inputTensor,
                                IntArrayRef sizes) {
  // In torch, -1 is passed for dimensions which are to stay the same size
  NSMutableArray<NSNumber*>* mpsSizes = [NSMutableArray array];
  [mpsSizes addObjectsFromArray:getMPSShape(sizes)];
  for (int64_t i = 0; i < mpsSizes.count; i++) {
    if ([mpsSizes[i] isEqualToNumber:[NSNumber numberWithInt:-1]]) {
      mpsSizes[i] = inputTensor.shape[i];
    }
  }
  return [mpsGraph broadcastTensor:inputTensor
                           toShape:mpsSizes
                              name:@"expand_copy"];
}

PyMPSGraphTensor*
MPSGraphModule::select(MPSGraphTensor* inputTensor, int dim, int index) {
  // Support python-style negative indexing
  // MPSGraph already handles negative indexing for start param
  if (dim < 0) {
    dim += inputTensor.shape.count;
  }
  MPSGraphTensor* slicedTensor =  [mpsGraph sliceTensor:inputTensor
                                              dimension:dim
                                                  start:index
                                                 length:1
                                                   name:@"slice"];
  slicedTensor = [mpsGraph squeezeTensor:slicedTensor
                                    axis:dim
                                    name:@"slice/squeezed"];
  return slicedTensor;
}

PyMPSGraphTensor*
MPSGraphModule::view(MPSGraphTensor* inputTensor,
                          IntArrayRef shape) {
  // MPS_TODO: Implement view functionality instead of just copying & reshaping
  return [mpsGraph reshapeTensor:inputTensor
                       withShape:getMPSShape(shape)
                            name:@"view_copy"];
}

PyMPSGraphTensor*
MPSGraphModule::permute(MPSGraphTensor* inputTensor,
                        IntArrayRef axes) {
  NSMutableArray<NSNumber*>* permutation = [NSMutableArray array];
  for(int64_t i = 0; i<axes.size(); i++) {
    [permutation addObject:[NSNumber numberWithInteger:axes[i]]];
  }
  return [mpsGraph transposeTensor:inputTensor
                       permutation:permutation
                              name:@"permutation"];
}

PyMPSGraphTensor*
MPSGraphModule::squeeze(MPSGraphTensor* inputTensor) {
  return [mpsGraph squeezeTensor:inputTensor
                            name:@"squeeze"];
}

PyMPSGraphTensor*
MPSGraphModule::squeeze(MPSGraphTensor* inputTensor, int dimension) {
  dimension = torch::maybe_wrap_dim(dimension, inputTensor.shape.count);
  TORCH_CHECK([inputTensor.shape[dimension] intValue] == 1, "Dimension must be size 1");
  return [mpsGraph squeezeTensor:inputTensor
                            axis:dimension
                            name:@"squeeze"];
}

PyMPSGraphTensor*
MPSGraphModule::squeeze(MPSGraphTensor* inputTensor, IntArrayRef axes) {
  NSMutableArray<NSNumber*>* wrappedAxes = [NSMutableArray array];
  for(int64_t i = 0; i<axes.size(); i++) {
    int dimension = torch::maybe_wrap_dim(axes[i], inputTensor.shape.count);
    TORCH_CHECK([inputTensor.shape[dimension] intValue] == 1, "Dimension must be size 1");
    [wrappedAxes addObject:[NSNumber numberWithInteger:dimension]];
  }
  return [mpsGraph squeezeTensor:inputTensor
                            axes:wrappedAxes
                            name:@"squeeze"];
}

PyMPSGraphTensor*
MPSGraphModule::unsqueeze(MPSGraphTensor* inputTensor, int dimension) {
  return [mpsGraph expandDimsOfTensor:inputTensor
                            axis:dimension
                            name:@"unsqueeze"];
}

}//namespace mps
