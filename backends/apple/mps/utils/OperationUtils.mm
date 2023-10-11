//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "OperationUtils.h"

namespace mps {

MPSShape* getMPSShape(const Tensor& t, MemoryFormat memory_format) {
  return getMPSShape(t.sizes(), memory_format);
}

MPSShape* getMPSShape(const IntArrayRef& sizes, MemoryFormat memory_format) {
  if (memory_format == MemoryFormat::ChannelsLast) {
    TORCH_INTERNAL_ASSERT(sizes.size() == 4, "ChannelsLast memory format must have 4 dimensions!");
    const NSUInteger N = sizes[0];
    const NSUInteger C = sizes[1];
    const NSUInteger H = sizes[2];
    const NSUInteger W = sizes[3];
    return @[@(N), @(H), @(W), @(C)];
  }
  const int sz = sizes.size();
  const int sz_ = (sz > 0) ? sz : 1;

  std::vector<NSNumber*> numbers(sz_);

  for (int i = 0; i < sz_; i++) {
    NSInteger sz_i = (i < sz) ? sizes[i] : 1;
    NSNumber* number = [NSNumber numberWithInteger:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers.data() count:numbers.size()];
}

std::vector<int64_t> getMPSShapeVec(const MPSShape* shape) {
  __block std::vector<int64_t> shapeVec;
  shapeVec.reserve([shape count]);
  [shape enumerateObjectsUsingBlock:^(NSNumber * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
      shapeVec.push_back(obj.intValue);
  }];
  return shapeVec;
}

MPSDataType getMPSScalarType(ScalarType scalar_type) {
  switch (scalar_type) {
    // This is an intentional fallthrough supporting Double for Scalar
    // types as they are casted to Float32 currently.
    case ScalarType::Double:
    case ScalarType::Float:
      return MPSDataTypeFloat32;
    case ScalarType::Half:  return MPSDataTypeFloat16;
    case ScalarType::Int:   return MPSDataTypeInt32;
    case ScalarType::Long:  return MPSDataTypeInt64;
    case ScalarType::Short: return MPSDataTypeInt16;
    case ScalarType::Char:
    case ScalarType::QInt8:
      return MPSDataTypeInt8;
    case ScalarType::Byte:
    case ScalarType::QUInt8:
      return MPSDataTypeUInt8;
    case ScalarType::Bool: return MPSDataTypeBool;
    default:
      TORCH_CHECK_TYPE(false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

MPSDataType getMPSDataType(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float: return MPSDataTypeFloat32;
    case ScalarType::Half:  return MPSDataTypeFloat16;
    case ScalarType::Int:   return MPSDataTypeInt32;
    case ScalarType::Long:  return MPSDataTypeInt64;
    case ScalarType::Short: return MPSDataTypeInt16;
    case ScalarType::Char:
    case ScalarType::QInt8:
      return MPSDataTypeInt8;
    case ScalarType::Byte:
    case ScalarType::QUInt8: return MPSDataTypeUInt8;
    case ScalarType::Bool:   return MPSDataTypeBool;
    case ScalarType::Double:
      TORCH_CHECK_TYPE(false, "Cannot convert a float64 Tensor to MPS as the MPS framework doesn't support float64. "
                       "Please use float32 instead.")
    default:
      TORCH_CHECK_TYPE(false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

ScalarType getScalarType(MPSDataType mps_data_type) {
 switch (mps_data_type) {
    case MPSDataTypeFloat32: return ScalarType::Float;
    case MPSDataTypeFloat16: return ScalarType::Half;
    case MPSDataTypeInt32:   return ScalarType::Int;
    case MPSDataTypeInt64:   return ScalarType::Long;
    case MPSDataTypeInt16:   return ScalarType::Short;
    case MPSDataTypeInt8:    return ScalarType::Char;
    case MPSDataTypeUInt8:   return ScalarType::Byte;
    case MPSDataTypeBool:    return ScalarType::Bool;
    default:
      TORCH_CHECK_TYPE(false, "Couldn't convert MPS data type ", mps_data_type, " to PyTorch data type");
  }
}

MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, MPSDataType toType) {
  return [mpsGraph castTensor:tensor toType:toType name:@"castTensor"];
}

MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, ScalarType toType) {
  return [mpsGraph castTensor:tensor toType:getMPSScalarType(toType) name:@"castTensor"];
}

MPSGraphTensorData* allocMPSGraphTensorData(id<MTLBuffer> buffer,
                                            MPSShape* mpsShape,
                                            MPSDataType mpsDataType) {
  MPSGraphTensorData *tensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                            shape:mpsShape
                                                                         dataType:mpsDataType] autorelease];
  TORCH_INTERNAL_ASSERT(tensorData);
  return tensorData;
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& src, MPSShape *mpsShape, MPSDataType dataType) : _tensor(src) {
  TORCH_CHECK(src.is_mps(), "Placeholder storage has not been allocated on MPS device!");
  // extract the pointer to MTLBuffer from the Tensor's storage
  id<MTLBuffer> srcBuf = getMTLBufferStorage(src);

  // tensor.numel() could be zero, but tensor is valid as long as the buffer size is non-zero.
  // if buffer size is zero in here, it's not a user error. It could be a missing check for
  // tensor.numel() == 0 in our internal implementations of ops.
  TORCH_INTERNAL_ASSERT([srcBuf length] > 0, "Placeholder tensor is empty!");
  const MPSDataType mpsDataType = dataType != MPSDataTypeInvalid ? dataType :
                      _tensor.dim() == 0 ? getMPSScalarType(_tensor.scalar_type()) : getMPSDataType(_tensor.scalar_type());

  if (!mpsShape) {
    mpsShape = getMPSShape(_tensor);
  }

  _value = allocMPSGraphTensorData(srcBuf, mpsShape, mpsDataType);
  TORCH_INTERNAL_ASSERT(_value);

  _placeholder = mpsGraphTensor;
}

} // namespace mps
