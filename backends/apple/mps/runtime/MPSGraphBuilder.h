//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

// Obj-C headers
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Runtime headers
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

// MPS headers
#include <executorch/backends/apple/mps/runtime/operations/MPSGraphSequoiaOps.h>
#include <executorch/backends/apple/mps/runtime/operations/MPSGraphVenturaOps.h>
#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <executorch/backends/apple/mps/schema_generated.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

using DataType = mpsgraph::MPSDataType;
using TensorPtr = const mpsgraph::MPSTensor *;
using NodePtr = const mpsgraph::MPSNode *;

#define _DEFINE_MPS_OP(name) Error mps##name##Op(NodePtr nodePtr);

/**
 * Helper class to construct a MPSGraph object from a serialized MPS FlatBuffer model.
 * It records all the input placeholders, lifted weights/biases and output feeds.
 */
class MPSGraphBuilder {
public:
  MPSGraphBuilder(const void *buffer_pointer, size_t num_bytes,
                  std::unordered_map<MPSGraphTensor *, int32_t> &mpsGraphTensorToId);
  ~MPSGraphBuilder() = default;

  Error compileModel();
  MPSGraph *getMPSGraph();
  MPSGraphExecutable *getMPSGraphExecutable();

private:
  // Input feeds & constant ops
  Error mpsGraphRankedPlaceholder(int32_t id);
  Error mpsConstantOp(int32_t id);
  // Activation ops
  _DEFINE_MPS_OP(HardTanh);
  _DEFINE_MPS_OP(ReLU);
  _DEFINE_MPS_OP(GELU);
  _DEFINE_MPS_OP(LeakyReLU);
  _DEFINE_MPS_OP(Softmax);
  _DEFINE_MPS_OP(LogSoftmax);
  // Arithmetic Binary Ops
  _DEFINE_MPS_OP(Add);
  _DEFINE_MPS_OP(Sub);
  _DEFINE_MPS_OP(Mul);
  _DEFINE_MPS_OP(Div);
  _DEFINE_MPS_OP(Pow);
  _DEFINE_MPS_OP(Fmod);
  _DEFINE_MPS_OP(Remainder);
  _DEFINE_MPS_OP(BitwiseAnd);
  _DEFINE_MPS_OP(BitwiseOr);
  _DEFINE_MPS_OP(BitwiseXor);
  _DEFINE_MPS_OP(Minimum);
  // Comparison ops
  _DEFINE_MPS_OP(Eq);
  _DEFINE_MPS_OP(Ne);
  _DEFINE_MPS_OP(Ge);
  _DEFINE_MPS_OP(Gt);
  _DEFINE_MPS_OP(Le);
  _DEFINE_MPS_OP(Lt);
  // Unary ops
  _DEFINE_MPS_OP(Exp);
  _DEFINE_MPS_OP(Exp2);
  _DEFINE_MPS_OP(Reciprocal);
  _DEFINE_MPS_OP(Sqrt);
  _DEFINE_MPS_OP(Neg);
  _DEFINE_MPS_OP(Log);
  _DEFINE_MPS_OP(Log10);
  _DEFINE_MPS_OP(Log2);
  _DEFINE_MPS_OP(Erf);
  _DEFINE_MPS_OP(Floor);
  _DEFINE_MPS_OP(Ceil);
  _DEFINE_MPS_OP(Rsqrt);
  _DEFINE_MPS_OP(Sigmoid);
  _DEFINE_MPS_OP(Sin);
  _DEFINE_MPS_OP(Sign);
  _DEFINE_MPS_OP(Cos);
  _DEFINE_MPS_OP(Tan);
  _DEFINE_MPS_OP(Abs);
  _DEFINE_MPS_OP(Asin);
  _DEFINE_MPS_OP(Acos);
  _DEFINE_MPS_OP(Atan);
  _DEFINE_MPS_OP(Sinh);
  _DEFINE_MPS_OP(Cosh);
  _DEFINE_MPS_OP(Tanh);
  _DEFINE_MPS_OP(Asinh);
  _DEFINE_MPS_OP(Acosh);
  _DEFINE_MPS_OP(Atanh);
  _DEFINE_MPS_OP(BitwiseNot);
  _DEFINE_MPS_OP(Isnan);
  _DEFINE_MPS_OP(Isinf);
  _DEFINE_MPS_OP(Round);
  _DEFINE_MPS_OP(LogicalNot);
  _DEFINE_MPS_OP(NormCdf);
  // Clamp ops
  _DEFINE_MPS_OP(Clamp);
  _DEFINE_MPS_OP(Where);
  // BitWise ops
  // Convolution ops
  _DEFINE_MPS_OP(Conv2D);
  _DEFINE_MPS_OP(DepthwiseConv2D);
  // Indexing ops
  _DEFINE_MPS_OP(IndexSelect);
  _DEFINE_MPS_OP(Embedding);
  _DEFINE_MPS_OP(IndexTensor);
  _DEFINE_MPS_OP(IndexPut);
  _DEFINE_MPS_OP(Scatter);
  // Linear algebra ops
  _DEFINE_MPS_OP(MatMul);
  _DEFINE_MPS_OP(Addmm);
  // Constant ops
  _DEFINE_MPS_OP(Full);
  _DEFINE_MPS_OP(FullLike);
  // Normalization ops
  _DEFINE_MPS_OP(BatchNorm);
  _DEFINE_MPS_OP(LayerNorm);
  // Reduce ops
  _DEFINE_MPS_OP(Mean);
  // Shape ops
  _DEFINE_MPS_OP(Permute);
  _DEFINE_MPS_OP(View);
  _DEFINE_MPS_OP(Expand);
  _DEFINE_MPS_OP(Cat);
  _DEFINE_MPS_OP(Squeeze);
  _DEFINE_MPS_OP(Unsqueeze);
  _DEFINE_MPS_OP(Select);
  _DEFINE_MPS_OP(Slice);
  _DEFINE_MPS_OP(PixelShuffle);
  _DEFINE_MPS_OP(SplitWithSizes);
  _DEFINE_MPS_OP(Cast);
  // Pooling ops
  _DEFINE_MPS_OP(MaxPool2DWithIndices);
  _DEFINE_MPS_OP(AvgPool2D);
  // Pad ops
  _DEFINE_MPS_OP(ConstantPadND);
  // Range ops
  _DEFINE_MPS_OP(Arange);
  // Quant-Dequant ops
  _DEFINE_MPS_OP(DequantizePerChannelGroup);

  // Helper functions
  Error addNodeToMPSGraph(NodePtr nodePtr);
  Error compileMetalKernel(NodePtr nodePtr);
  MPSShape *getMPSShape(int32_t id);
  MPSShape *getMPSShape(const flatbuffers::Vector<int32_t> *shape);
  int64_t numel(const flatbuffers::Vector<int32_t> *shape);
  MPSDataType getMPSDataType(int32_t id);
  MPSDataType getMPSDataType(DataType serializedDataType);
  MPSGraphTensor *getMPSGraphTensor(int32_t id);
  NSData *getConstantData(int32_t id);
  std::pair<float, float> getMinMaxValues(NodePtr nodePtr);
  Error compileMPSGraph();
  Error compileMetalKernel();

  // Each MPSGraph op result in at least MPSGraphTensor being
  // produced, which will be stored in this structure. Other ops
  // can reference the saved tensor by the AOT id (1:1 mapping).
  std::vector<MPSGraphTensor *> _idToMPSGraphTensor;
  std::unordered_map<MPSGraphTensor *, int32_t> &_mpsGraphTensorToId;
  // FlatBuffer serialized graph containing the nodes from the original model.
  const mpsgraph::MPSGraph *_flatBufferGraph;
  // FlatBuffer raw bytes of the serialized MPS model.
  const void *_buffer_pointer;
  size_t _num_bytes;

  bool _metal_kernel;
  MPSGraph *_mpsGraph;
  MPSGraphExecutable *_mpsGraphExecutable;
  NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *_feeds;
  NSMutableArray<MPSGraphTensor *> *_targetTensors;

  const uint8_t *_constant_data_ptr;
};

#undef _DEFINE_MPS_OP

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
