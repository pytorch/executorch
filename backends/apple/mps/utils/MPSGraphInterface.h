//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <torch/torch.h>
#include "OperationUtils.h"
#include "operations/BinaryOps.h"
#include "operations/UnaryOps.h"

// Workaround for PyBind custom class return type.
// We need a type caster
// (https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html) for
// MPSGraphTensor. Return `void*` for now instead of `MPSGraphTensor*`.
typedef void PyMPSGraphTensor;

namespace mps {

using namespace torch;

// ExecuTorch is supported only from macOS 14.0 and above
// Previous macOS version don't have support to generate .mpsgraphpackage
enum class MacOSVersion : uint32_t {
  MACOS_VER_14_0_PLUS = 0,
};

class MPSGraphModule {
 public:
  MPSGraphModule();
  ~MPSGraphModule();

  // Graph placeholders.
  PyMPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSDataType dataType);
  PyMPSGraphTensor* mpsGraphRankedPlaceHolder(
      MPSDataType dataType,
      const IntArrayRef& shape);
  PyMPSGraphTensor* mpsGraphScalarPlaceHolder(MPSDataType dataType);
  void set_outputs(py::args args);

  // Graph operators.
  PyMPSGraphTensor* constant(double scalar, MPSDataType dataType);
  PyMPSGraphTensor* constantTensor(
      Tensor constant_tensor,
      MPSDataType dataType);
  PyMPSGraphTensor* constantWithScalar(
      MPSDataType dtype,
      const IntArrayRef& sizes,
      double scalar);
  PyMPSGraphTensor* full(IntArrayRef size, double scalar, MPSDataType dataType);
  PyMPSGraphTensor* full_like(MPSGraphTensor* inputTensor, double scalar);
  std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*> batchNorm(
      MPSGraphTensor* inputTensor,
      MPSGraphTensor* meanTensor,
      MPSGraphTensor* varTensor,
      MPSGraphTensor* weightTensor,
      MPSGraphTensor* biasTensor,
      float momentum,
      float epsilon);
  std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*> layerNorm(
      MPSGraphTensor* inputTensor,
      IntArrayRef normalized_shape,
      MPSGraphTensor* weightTensor,
      MPSGraphTensor* biasTensor,
      float epsilon);
  PyMPSGraphTensor* conv2D(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor,
      MPSGraphTensor* bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      bool transpose,
      IntArrayRef outputPadding,
      int64_t groups,
      bool is_depthwise);
  std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*> maxPool2DWithIndices(
      MPSGraphTensor* inputTensor,
      IntArrayRef kernel_size,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      bool ceil_mode);
  PyMPSGraphTensor* avgPool2D(
      MPSGraphTensor* inputTensor,
      IntArrayRef kernel_size,
      IntArrayRef stride,
      IntArrayRef padding,
      bool ceil_mode,
      bool count_include_pad,
      c10::optional<int> divisor_override);
  PyMPSGraphTensor*
  hardTanh(MPSGraphTensor* inputTensor, float min_val, float max_val);
  PyMPSGraphTensor*
  mean(MPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims);
  std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*>
  minDim(MPSGraphTensor* inputTensor, int dim, bool keep_dims);
  std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*>
  maxDim(MPSGraphTensor* inputTensor, int dim, bool keep_dims);
  PyMPSGraphTensor*
  amax(MPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims);
  PyMPSGraphTensor*
  amin(MPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims);
  PyMPSGraphTensor* argmax(
      MPSGraphTensor* inputTensor,
      int64_t dim,
      bool keep_dims,
      bool flatten);
  PyMPSGraphTensor* argmin(
      MPSGraphTensor* inputTensor,
      int64_t dim,
      bool keep_dims,
      bool flatten);
  PyMPSGraphTensor* mm(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor);
  PyMPSGraphTensor* identity(MPSGraphTensor* inputTensor);
  PyMPSGraphTensor* clamp(
      MPSGraphTensor* inputTensor,
      float min,
      float max,
      bool use_min,
      bool use_max);
  PyMPSGraphTensor* relu(MPSGraphTensor* inputTensor);
  PyMPSGraphTensor* leaky_relu(
      MPSGraphTensor* inputTensor,
      float negative_slope);
  PyMPSGraphTensor*
  softmax(MPSGraphTensor* inputTensor, int dim, bool half_to_float);
  PyMPSGraphTensor*
  log_softmax(MPSGraphTensor* inputTensor, int dim, bool half_to_float);
  PyMPSGraphTensor* squeeze(MPSGraphTensor* inputTensor);
  PyMPSGraphTensor* squeeze(MPSGraphTensor* inputTensor, int dim);
  PyMPSGraphTensor* squeeze(MPSGraphTensor* inputTensor, IntArrayRef dim);
  PyMPSGraphTensor* unsqueeze(MPSGraphTensor* inputTensor, int dimension);
  PyMPSGraphTensor* gelu(
      MPSGraphTensor* inputTensor,
      const std::string& approximation);
  PyMPSGraphTensor* glu(MPSGraphTensor* inputTensor, int64_t dim);
  PyMPSGraphTensor* cat(int dim, py::args catTensors);
  PyMPSGraphTensor* pixel_shuffle(
      MPSGraphTensor* inputTensor,
      int upscale_factor);
  std::vector<PyMPSGraphTensor*>
  split(MPSGraphTensor* inputTensor, int split_size, int dim);
  std::vector<PyMPSGraphTensor*>
  split_size(MPSGraphTensor* inputTensor, IntArrayRef split_sizes, int dim);
  std::vector<PyMPSGraphTensor*> unbind(MPSGraphTensor* inputTensor, int dim);
  PyMPSGraphTensor* stack(int dim, py::args stackTensors);
  PyMPSGraphTensor* slice(
      MPSGraphTensor* inputTensor,
      int64_t dim,
      c10::optional<int64_t> start,
      c10::optional<int64_t> end,
      int64_t step);
  PyMPSGraphTensor* expand(MPSGraphTensor* inputTensor, IntArrayRef sizes);
  PyMPSGraphTensor* select(MPSGraphTensor* inputTensor, int dim, int index);
  PyMPSGraphTensor* view(MPSGraphTensor* inputTensor, IntArrayRef shape);
  PyMPSGraphTensor* permute(MPSGraphTensor* inputTensor, IntArrayRef axes);
  PyMPSGraphTensor* cumsum(MPSGraphTensor* inputTensor, int dim);
  PyMPSGraphTensor* addmm(
      MPSGraphTensor* biasTensor,
      MPSGraphTensor* inputTensor,
      MPSGraphTensor* weightTensor,
      float beta,
      float alpha);
  MPSGraphTensor* trunc_tensor(MPSGraphTensor* inputTensor);

  // Binary Ops
  PyMPSGraphTensor* div_mode_template(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor,
      c10::optional<c10::string_view> rounding_mode,
      const string& op_name);
  PyMPSGraphTensor* binaryOpTensor(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor,
      const std::string& op_name,
      std::function<MPSGraphTensor*(MPSGraphTensor*, MPSGraphTensor*)>
          binaryOpFunction);

  PyMPSGraphTensor* binaryOpWithScalar(
      MPSGraphTensor* inputTensor,
      Scalar scalar,
      const std::string& op_name,
      std::function<MPSGraphTensor*(MPSGraphTensor*, MPSGraphTensor*)>
          binaryOpFunction);

  // Unary Ops
  PyMPSGraphTensor* unaryOpTensor(
      MPSGraphTensor* inputTensor,
      const std::string& op_name,
      std::function<MPSGraphTensor*(MPSGraphTensor*)> unaryOpFunction);

  PyMPSGraphTensor* additionWithTensor(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor,
      Scalar alpha);
  PyMPSGraphTensor* subtractionWithTensor(
      MPSGraphTensor* primaryTensor,
      MPSGraphTensor* secondaryTensor,
      Scalar alpha);
  PyMPSGraphTensor* multiplicationWithScalar(
      MPSGraphTensor* inputTensor,
      Scalar scalar);

  // bitwise Ops
  PyMPSGraphTensor* bitwiseNotTensor(
      MPSGraphTensor* inputTensor,
      const std::string& op_name);

  // Pad Ops
  PyMPSGraphTensor*
  constant_pad_nd(MPSGraphTensor* input, IntArrayRef pad, const double value);

  // range Ops
  PyMPSGraphTensor* arange(
      Scalar start,
      Scalar end,
      Scalar step,
      MPSDataType dataType,
      const int numEle);

  // trinary Ops
  PyMPSGraphTensor*
  where(MPSGraphTensor* cond, MPSGraphTensor* input, MPSGraphTensor* other);

  // Indexing Ops
  PyMPSGraphTensor* index_select(
      MPSGraphTensor* inputTensor,
      int64_t dim,
      MPSGraphTensor* indexTensor);

  MPSGraph* getMPSGraph() {
    return mpsGraph;
  }

  // Graph debug methods.
  void printGraph();
  bool macos_version_or_newer(
      MacOSVersion version = MacOSVersion::MACOS_VER_14_0_PLUS);

  MPSGraphExecutable* compileMPSGraphExecutable();
  std::vector<uint8_t> serialize();

 private:
  MPSGraph* mpsGraph;
  std::vector<MPSGraphTensor*> outputTensors_;
  std::vector<MPSGraphTensor*> inputTensors_;
  MPSGraphExecutable* executable_;
};

} // namespace mps
