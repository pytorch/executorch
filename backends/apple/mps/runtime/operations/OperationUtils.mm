
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <numeric>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSDataType
MPSGraphBuilder::getMPSDataType(int32_t id) {
  return getMPSDataType(_flatBufferGraph->mps_values()->Get(id)->datatype());
}

MPSDataType
MPSGraphBuilder::getMPSDataType(DataType serializedDataType) {
  switch (serializedDataType) {
    case DataType::mps_data_type_float16:
      return MPSDataTypeFloat16;
    case DataType::mps_data_type_float32:
    case DataType::mps_data_type_float64:
      return MPSDataTypeFloat32;
    case DataType::mps_data_type_int8:
      return MPSDataTypeInt8;
    case DataType::mps_data_type_int4: {
      if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
        return MPSDataTypeInt4;
      } else {
        return ((MPSDataType)(MPSDataTypeSignedBit | 4));
      }
    }
    case DataType::mps_data_type_int16:
      return MPSDataTypeInt16;
    case DataType::mps_data_type_int32:
      return MPSDataTypeInt32;
    case DataType::mps_data_type_int64:
      return MPSDataTypeInt64;
    case DataType::mps_data_type_bool:
      return MPSDataTypeBool;
    default:
      ET_CHECK_MSG(false, "[ERROR] Invalid MPS data type: %d!", (int32_t)serializedDataType);
      return MPSDataTypeInvalid;
  }
}

MPSShape*
MPSGraphBuilder::getMPSShape(int32_t id) {
  TensorPtr mpsTensor = _flatBufferGraph->mps_values()->Get(id);
  auto sizes = mpsTensor->dims();
  const int sz =  mpsTensor->num_dims();
  const int sz_ = (sz > 0) ? sz : 1;

  std::vector<NSNumber*> numbers(sz_);

  for (int i = 0; i < sz_; i++) {
    NSInteger sz_i = (i < sz) ? sizes->Get(i) : 1;
    NSNumber* number = [NSNumber numberWithInteger:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers.data() count:numbers.size()];
}

MPSShape*
MPSGraphBuilder::getMPSShape(const flatbuffers::Vector<int32_t>* shape) {
  const int sz =  shape->size();
  const int sz_ = (sz > 0) ? sz : 1;

  std::vector<NSNumber*> numbers(sz_);

  for (int i = 0; i < sz_; i++) {
    NSInteger sz_i = (i < sz) ? shape->Get(i) : 1;
    NSNumber* number = [NSNumber numberWithInteger:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers.data() count:numbers.size()];
}

int64_t
MPSGraphBuilder::numel(const flatbuffers::Vector<int32_t>* shape) {
  int64_t numel = 1;
  for (auto dim : *shape) {
    numel = numel * dim;
  }
  return numel;
}

NSData*
MPSGraphBuilder::getConstantData(int32_t id) {
  TensorPtr mpsTensor = _flatBufferGraph->mps_values()->Get(id);
  uint64_t constantBufferSize = mpsTensor->constant_buffer_size();
  uint64_t segmentOffset = mpsTensor->segment_offset();
  const unsigned char* constantBuffer = _constant_data_ptr + segmentOffset;
  ET_CHECK_MSG(constantBufferSize > 0 && constantBuffer != nullptr, "[ERROR] Invalid constant buffer");
  return [[NSData alloc] initWithBytesNoCopy:(void*)constantBuffer
                                length:constantBufferSize];
}

std::pair<float, float>
MPSGraphBuilder::getMinMaxValues(NodePtr nodePtr) {
  float minValue = -INF;
  float maxValue = INF;
  auto minMaxValues = nodePtr->min_max();
  if (minMaxValues != nullptr) {
    minValue = minMaxValues->min_value();
    maxValue = minMaxValues->max_value();
  }

  return {minValue, maxValue};
}

#define _DEFINE_MPS_NODE(node)                 \
  case mpsgraph::MPSNodeUnion::MPS##node:      \
    return mps##node##Op(nodePtr);

Error
MPSGraphBuilder::addNodeToMPSGraph(NodePtr nodePtr) {
  switch (nodePtr->mpsnode_union_type()) {
    // Activation ops
    _DEFINE_MPS_NODE(HardTanh);
    _DEFINE_MPS_NODE(ReLU);
    _DEFINE_MPS_NODE(GELU);
    _DEFINE_MPS_NODE(LeakyReLU);
    _DEFINE_MPS_NODE(Softmax);
    _DEFINE_MPS_NODE(LogSoftmax);
    // Binary ops
    _DEFINE_MPS_NODE(Add);
    _DEFINE_MPS_NODE(Sub);
    _DEFINE_MPS_NODE(Mul);
    _DEFINE_MPS_NODE(Div);
    _DEFINE_MPS_NODE(Pow);
    _DEFINE_MPS_NODE(Fmod);
    _DEFINE_MPS_NODE(Remainder);
    _DEFINE_MPS_NODE(BitwiseAnd);
    _DEFINE_MPS_NODE(BitwiseOr);
    _DEFINE_MPS_NODE(BitwiseXor);
    _DEFINE_MPS_NODE(Minimum);
    // Unary ops
    _DEFINE_MPS_NODE(Exp);
    _DEFINE_MPS_NODE(Exp2);
    _DEFINE_MPS_NODE(Reciprocal);
    _DEFINE_MPS_NODE(Sqrt);
    _DEFINE_MPS_NODE(Neg);
    _DEFINE_MPS_NODE(Log);
    _DEFINE_MPS_NODE(Log10);
    _DEFINE_MPS_NODE(Log2);
    _DEFINE_MPS_NODE(Erf);
    _DEFINE_MPS_NODE(Floor);
    _DEFINE_MPS_NODE(Ceil);
    _DEFINE_MPS_NODE(Rsqrt);
    _DEFINE_MPS_NODE(Sigmoid);
    _DEFINE_MPS_NODE(Sin);
    _DEFINE_MPS_NODE(Sign);
    _DEFINE_MPS_NODE(Cos);
    _DEFINE_MPS_NODE(Tan);
    _DEFINE_MPS_NODE(Abs);
    _DEFINE_MPS_NODE(Asin);
    _DEFINE_MPS_NODE(Acos);
    _DEFINE_MPS_NODE(Atan);
    _DEFINE_MPS_NODE(Sinh);
    _DEFINE_MPS_NODE(Cosh);
    _DEFINE_MPS_NODE(Tanh);
    _DEFINE_MPS_NODE(Asinh);
    _DEFINE_MPS_NODE(Acosh);
    _DEFINE_MPS_NODE(Atanh);
    _DEFINE_MPS_NODE(BitwiseNot);
    _DEFINE_MPS_NODE(Isnan);
    _DEFINE_MPS_NODE(Isinf);
    _DEFINE_MPS_NODE(Round);
    _DEFINE_MPS_NODE(LogicalNot);
    // Clamp ops
    _DEFINE_MPS_NODE(Clamp);
    _DEFINE_MPS_NODE(Where);
    // Linear algebra ops
    _DEFINE_MPS_NODE(MatMul);
    _DEFINE_MPS_NODE(Addmm);
    // Constant ops
    _DEFINE_MPS_NODE(Full);
    _DEFINE_MPS_NODE(FullLike);
    //Indexing ops
    _DEFINE_MPS_NODE(IndexSelect);
    _DEFINE_MPS_NODE(Embedding);
    _DEFINE_MPS_NODE(IndexTensor);
    _DEFINE_MPS_NODE(IndexPut);
    _DEFINE_MPS_NODE(Scatter);
    // Reduce ops
    _DEFINE_MPS_NODE(Mean);
    // Shape ops
    _DEFINE_MPS_NODE(Permute);
    _DEFINE_MPS_NODE(View);
    _DEFINE_MPS_NODE(Expand);
    _DEFINE_MPS_NODE(Cat);
    _DEFINE_MPS_NODE(Squeeze);
    _DEFINE_MPS_NODE(Unsqueeze);
    _DEFINE_MPS_NODE(Select);
    _DEFINE_MPS_NODE(Slice);
    _DEFINE_MPS_NODE(PixelShuffle);
    _DEFINE_MPS_NODE(SplitWithSizes);
    _DEFINE_MPS_NODE(Cast);
    // Convolution ops
    _DEFINE_MPS_NODE(Conv2D);
    _DEFINE_MPS_NODE(DepthwiseConv2D);
    // Comparison ops
    _DEFINE_MPS_NODE(Eq);
    _DEFINE_MPS_NODE(Ne);
    _DEFINE_MPS_NODE(Ge);
    _DEFINE_MPS_NODE(Gt);
    _DEFINE_MPS_NODE(Le);
    _DEFINE_MPS_NODE(Lt);
    // Normalization ops
    _DEFINE_MPS_NODE(BatchNorm);
    _DEFINE_MPS_NODE(LayerNorm);
    // Pooling ops
    _DEFINE_MPS_NODE(MaxPool2DWithIndices);
    _DEFINE_MPS_NODE(AvgPool2D);
    // Pad ops
    _DEFINE_MPS_NODE(ConstantPadND);
    // Range ops
    _DEFINE_MPS_NODE(Arange);
    // Quant-Dequant ops
    _DEFINE_MPS_NODE(DequantizePerChannelGroup);

    case mpsgraph::MPSNodeUnion::NONE:
    default:
      ET_CHECK_OR_RETURN_ERROR(
        false,
        NotImplemented,
        "[ERROR] Unhandled node type: %s!",
        mpsgraph::EnumNameMPSNodeUnion(nodePtr->mpsnode_union_type()));
  }
}

Error
MPSGraphBuilder::compileMetalKernel(NodePtr nodePtr) {
  return addNodeToMPSGraph(nodePtr);
}

#undef _DEFINE_MPS_NODE

MPSGraphTensor*
MPSGraphBuilder::getMPSGraphTensor(int32_t id) {
  return _idToMPSGraphTensor[id];
}

MPSDataType getMPSScalarType(exec_aten::ScalarType scalar_type) {
  switch (scalar_type) {
    // This is an intentional fallthrough supporting Double for Scalar
    // types as they are casted to Float32 currently.
    case exec_aten::ScalarType::Float:
      return MPSDataTypeFloat32;
    case exec_aten::ScalarType::Half:
      return MPSDataTypeFloat16;
    default:
      ET_CHECK_MSG(false, "Unhandled ExecuTorch scalar type!");
  }
}

exec_aten::ScalarType getScalarType(MPSDataType mpsDataType) {
  switch (mpsDataType) {
    case MPSDataTypeFloat16:
      return exec_aten::ScalarType::Half;
    case MPSDataTypeFloat32:
      return exec_aten::ScalarType::Float;
    case MPSDataTypeInt8:
      return exec_aten::ScalarType::Char;
    case MPSDataTypeInt16:
      return exec_aten::ScalarType::Short;
    case MPSDataTypeInt32:
      return exec_aten::ScalarType::Int;
    case MPSDataTypeInt64:
      return exec_aten::ScalarType::Long;
    case MPSDataTypeBool:
      return exec_aten::ScalarType::Bool;
    default:
      ET_CHECK_MSG(false, "Unhandled MPS data type!");
  }
}

MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, exec_aten::ScalarType toType) {
  return castMPSTensor(mpsGraph, tensor, getMPSScalarType(toType));
}

MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, MPSDataType toType) {
  return [mpsGraph castTensor:tensor toType:toType name:@"castTensor"];
}

std::vector<int64_t> getMPSShapeVec(const MPSShape* shape) {
  __block std::vector<int64_t> shapeVec;
  shapeVec.reserve([shape count]);
  [shape enumerateObjectsUsingBlock:^(NSNumber * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
      shapeVec.push_back(obj.intValue);
  }];
  return shapeVec;
}

id<MTLBuffer> getMTLBufferStorage(const Tensor &tensor) {
  uint8_t *data = tensor.mutable_data_ptr<uint8_t>();
  return [MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:data
                                                               length:tensor.nbytes()
                                                              options:0
                                                          deallocator:nil];
}

void* pageAlignedBlockPtr(const void* ptr, NSUInteger size, NSUInteger* alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  assert(address >= alignedAddress);
  assert(address + size <= alignedAddress + alignedLength);

  *alignedBlockSize = alignedLength;
  return (void*)alignedAddress;
}


MPSGraphTensor* permuteTensor(MPSGraph* graph, MPSGraphTensor* inputTensor, NSArray* permuteOrder) {
  NSUInteger srcRank = [[inputTensor shape] count];
  if (srcRank != [permuteOrder count]) {
    return nil;
  }

  MPSGraphTensor* outputTensor = inputTensor;
  std::vector<NSUInteger> dimensionOrder(srcRank);
  std::iota(std::begin(dimensionOrder), std::end(dimensionOrder), 0);

  for (int32_t i = 0; i < srcRank; i++) {
    NSUInteger axis = [permuteOrder[i] integerValue];
    auto axisIter = std::find(dimensionOrder.begin(), dimensionOrder.end(), axis);
    NSUInteger axis1 = i;
    NSUInteger axis2 = axisIter - dimensionOrder.begin();
    iter_swap(dimensionOrder.begin() + i, axisIter);

    outputTensor = [graph transposeTensor:outputTensor dimension:axis1 withDimension:axis2 name:nil];
  }

  return outputTensor;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
