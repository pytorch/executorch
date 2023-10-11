//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "MPSGraphInterface.h"

namespace mps {
namespace {
using namespace torch;
// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Main class to interface with MPSGraph.
  py::class_<MPSGraphModule>(m, "MPSGraphModule")
    // MPSGraphModule constructor.
    .def(py::init<>())

    //
    // Graph placeholders.
    //
    .def("mpsGraphUnrankedPlaceHolder", &MPSGraphModule::mpsGraphUnrankedPlaceHolder)
    .def("mpsGraphRankedPlaceHolder",   &MPSGraphModule::mpsGraphRankedPlaceHolder)
    .def("mpsGraphScalarPlaceHolder",   &MPSGraphModule::mpsGraphScalarPlaceHolder)
    .def("set_outputs",  &MPSGraphModule::set_outputs)

    //
    // Graph operators
    //
    .def("constant", &MPSGraphModule::constant)
    .def("constantTensor", &MPSGraphModule::constantTensor)
    .def("full", &MPSGraphModule::full)
    .def("full_like", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, double scalar) {
      return  self.full_like(static_cast<MPSGraphTensor*>(inputTensor), scalar);
    })
    .def("mm", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor) {
      return  self.mm(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor));
    })
    .def("conv2D", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor,
                      std::optional<PyMPSGraphTensor*> biasTensor, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
                      bool transposed, IntArrayRef outputPadding, int64_t groups, bool is_depthwise) {
      MPSGraphTensor *optionalBias = nullptr;
      MPSGraphTensor *inputTensor = static_cast<MPSGraphTensor*>(primaryTensor);
      if(biasTensor.has_value()){
        optionalBias = static_cast<MPSGraphTensor*>(*biasTensor);
      }
      return self.conv2D(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor),
                          optionalBias, stride, padding, dilation, transposed, outputPadding, groups, is_depthwise);
    })
    .def("maxPool2DWithIndices", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef kernel_size,
                                    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
      return self.maxPool2DWithIndices(static_cast<MPSGraphTensor*>(inputTensor), kernel_size, stride, padding,
                                      dilation, ceil_mode);
    })
    .def("avgPool2D", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef kernel_size,
                                    IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                                    c10::optional<int> divisor_override) {
      return self.avgPool2D(static_cast<MPSGraphTensor*>(inputTensor), kernel_size, stride, padding,
                                      ceil_mode, count_include_pad, divisor_override);

    })
    .def("batchNorm", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, PyMPSGraphTensor* weightTensor,
                          PyMPSGraphTensor* biasTensor, PyMPSGraphTensor* meanTensor, PyMPSGraphTensor* varTensor,
                          float momentum, float epsilon) {
      std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*> result = self.batchNorm(
                            static_cast<MPSGraphTensor*>(inputTensor),
                            static_cast<MPSGraphTensor*>(meanTensor),
                            static_cast<MPSGraphTensor*>(varTensor),
                            static_cast<MPSGraphTensor*>(weightTensor),
                            static_cast<MPSGraphTensor*>(biasTensor),
                            momentum, epsilon);
      return result;

    })
    .def("layerNorm", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef normalized_shape,
                          c10::optional<PyMPSGraphTensor*> weightTensor_opt, c10::optional<PyMPSGraphTensor*> biasTensor_opt,
                          float epsilon) {

      MPSGraphTensor* weightTensor = nil;
      MPSGraphTensor* biasTensor = nil;
      if(weightTensor_opt.has_value()){
        weightTensor = static_cast<MPSGraphTensor*>(*weightTensor_opt);
      }
      if(biasTensor_opt.has_value()){
        biasTensor = static_cast<MPSGraphTensor*>(*biasTensor_opt);
      }

      std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*> result = self.layerNorm(
        static_cast<MPSGraphTensor*>(inputTensor), normalized_shape,
        weightTensor, biasTensor, epsilon);

      return result;
    })
    .def("hardTanh", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, float min_value, float max_value) {
      return self.hardTanh(static_cast<MPSGraphTensor*>(inputTensor), min_value, max_value);
    })
    .def("mean", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims) {
      return self.mean(static_cast<MPSGraphTensor*>(inputTensor), dims, keep_dims);
    })
    .def("minDim", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim, bool keep_dims) {
      return self.minDim(static_cast<MPSGraphTensor*>(inputTensor), dim, keep_dims);
    })
    .def("maxDim", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim, bool keep_dims) {
      return self.maxDim(static_cast<MPSGraphTensor*>(inputTensor), dim, keep_dims);
    })
    .def("amax", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims) {
      return self.amax(static_cast<MPSGraphTensor*>(inputTensor), dims, keep_dims);
    })
    .def("amin", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef dims, bool keep_dims) {
      return self.amin(static_cast<MPSGraphTensor*>(inputTensor), dims, keep_dims);
    })
    .def("argmax", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int64_t dim, bool keep_dims, bool flatten) {
      return self.argmax(static_cast<MPSGraphTensor*>(inputTensor), dim, keep_dims, flatten);
    })
    .def("argmin", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int64_t dim, bool keep_dims, bool flatten) {
      return self.argmin(static_cast<MPSGraphTensor*>(inputTensor), dim, keep_dims, flatten);
    })
    .def("identity", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor) {
      return self.identity(static_cast<MPSGraphTensor*>(inputTensor));
    })
    .def("clamp", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, float min, float max, bool use_min, bool use_max) {
      return self.clamp(static_cast<MPSGraphTensor*>(inputTensor), min, max, use_min, use_max);
    })
    .def("relu", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor) {
      return self.relu(static_cast<MPSGraphTensor*>(inputTensor));
    })
    .def("leaky_relu", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, float negative_slope) {
      return self.leaky_relu(static_cast<MPSGraphTensor*>(inputTensor), negative_slope);
    })
    .def("softmax", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim, bool half_to_float) {
      return self.softmax(static_cast<MPSGraphTensor*>(inputTensor), dim, half_to_float);
    })
    .def("log_softmax", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim, bool half_to_float) {
      return self.log_softmax(static_cast<MPSGraphTensor*>(inputTensor), dim, half_to_float);
    })
    .def("squeeze", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor) {
      return self.squeeze(static_cast<MPSGraphTensor*>(inputTensor));
    })
    .def("squeeze", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim) {
      return self.squeeze(static_cast<MPSGraphTensor*>(inputTensor), dim);
    })
    .def("squeeze", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef dims) {
      return self.squeeze(static_cast<MPSGraphTensor*>(inputTensor), dims);
    })
    .def("unsqueeze", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dimension) {
      return self.unsqueeze(static_cast<MPSGraphTensor*>(inputTensor), dimension);
    })
    .def("gelu", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, const std::string &approximation) {
      return self.gelu(static_cast<MPSGraphTensor*>(inputTensor), approximation);
    })
    .def("glu", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int64_t dim) {
      return self.glu(static_cast<MPSGraphTensor*>(inputTensor), dim);
    })
    .def("pixel_shuffle", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int upscale_factor) {
      return self.pixel_shuffle(static_cast<MPSGraphTensor*>(inputTensor), upscale_factor);
    })
    .def("split_size", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef split_sizes, int dim) {
      return self.split_size(static_cast<MPSGraphTensor*>(inputTensor), split_sizes, dim);
    })
    .def("split", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int split_size, int dim) {
      return self.split(static_cast<MPSGraphTensor*>(inputTensor), split_size, dim);
    })
    .def("unbind", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim) {
      return self.unbind(static_cast<MPSGraphTensor*>(inputTensor), dim);
    })
    .def("cat", [](MPSGraphModule& self,int dim, py::args catTensors) {
      return self.cat(dim, catTensors);
    })
    .def("stack", [](MPSGraphModule& self,int dim, py::args stackTensors) {
      return self.stack(dim, stackTensors);
    })
    .def("slice", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
      return self.slice(static_cast<MPSGraphTensor*>(inputTensor), dim, start, end, step);
    })
    .def("expand", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef sizes){
      return self.expand(static_cast<MPSGraphTensor*>(inputTensor), sizes);
    })
    .def("select", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim, int index) {
      return self.select(static_cast<MPSGraphTensor*>(inputTensor), dim, index);
    })
    .def("view", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef shape){
      return self.view(static_cast<MPSGraphTensor*>(inputTensor), shape);
    })
    .def("permute", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef axes) {
      return self.permute(static_cast<MPSGraphTensor*>(inputTensor), axes);
    })
    .def("cumsum", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int dim) {
      return self.cumsum(static_cast<MPSGraphTensor*>(inputTensor), dim);
    })
    .def("addmm", [](MPSGraphModule& self, PyMPSGraphTensor* biasTensor,
                    PyMPSGraphTensor* inputTensor, PyMPSGraphTensor* weightTensor,
                    float beta, float alpha) {
      return self.addmm(static_cast<MPSGraphTensor*>(biasTensor),
                        static_cast<MPSGraphTensor*>(inputTensor),
                        static_cast<MPSGraphTensor*>(weightTensor),
                        beta, alpha);
    })
    .def("constant_pad_nd", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, IntArrayRef pad, const double value) {
      return self.constant_pad_nd(static_cast<MPSGraphTensor*>(inputTensor), pad, value);
    })
    .def("add", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor, float alpha) {
      return self.additionWithTensor(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), alpha);
    })
    .def("add", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor, int alpha) {
      return self.additionWithTensor(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), alpha);
    })
    .def("sub", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor, float alpha) {
      return self.subtractionWithTensor(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), alpha);
    })
    .def("sub", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor, int alpha) {
      return self.subtractionWithTensor(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), alpha);
    })
    .def("mulWithScalar", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, float scalar) {
      return self.multiplicationWithScalar(static_cast<MPSGraphTensor*>(inputTensor), scalar);
    })
    .def("mulWithScalar", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int scalar) {
      return self.multiplicationWithScalar(static_cast<MPSGraphTensor*>(inputTensor), scalar);
    })
    .def("arange", [](MPSGraphModule& self, int64_t start, int64_t end, int64_t step,
      MPSDataType dtype, int numOfElements) {
      return self.arange(start, end, step, dtype, numOfElements);
    })
    .def("arange", [](MPSGraphModule& self, float start, float end, float step,
      MPSDataType dtype, int numOfElements) {
      return self.arange(start, end, step, dtype, numOfElements);
    })
    .def("where", [](MPSGraphModule& self, PyMPSGraphTensor* cond, PyMPSGraphTensor* input,
       PyMPSGraphTensor* other) {
      return self.where(static_cast<MPSGraphTensor*>(cond), static_cast<MPSGraphTensor*>(input),
       static_cast<MPSGraphTensor*>(other));
    })
    .def("scalar_out", [](MPSGraphModule& self, double scalar, MPSDataType dtype) {
      return self.constant(scalar, dtype);
    })
    .def("index_select", [](MPSGraphModule& self, PyMPSGraphTensor* inputTensor, int64_t dim, PyMPSGraphTensor* indexTensor) {
      return self.index_select(static_cast<MPSGraphTensor*>(inputTensor), dim, static_cast<MPSGraphTensor*>(indexTensor));
    })
    .def("empty", [](MPSGraphModule& self, IntArrayRef sizes, MPSDataType dtype) {
      return self.constantWithScalar(dtype, sizes, 0);
    })
    // Arithmetic Binary Ops
    REGISTER_PYBIND11_MPS_BINARY_OP("add", addition)
    REGISTER_PYBIND11_MPS_BINARY_OP("sub", subtraction)
    REGISTER_PYBIND11_MPS_BINARY_OP("mul", multiplication)
    REGISTER_PYBIND11_MPS_BINARY_OP("min", minimum)
    REGISTER_PYBIND11_MPS_BINARY_OP("max", maximum)
    REGISTER_PYBIND11_MPS_BINARY_OP("pow", power)
    REGISTER_PYBIND11_MPS_BINARY_OP("remainder", modulo)
    REGISTER_PYBIND11_MPS_BINARY_OP("atan2", atan2)
    REGISTER_PYBIND11_MPS_BINARY_OP("bmm", matrixMultiplication)
    REGISTER_PYBIND11_MPS_BINARY_OP("minimum", minimum)

    // Comparison Ops
    REGISTER_PYBIND11_MPS_BINARY_OP("eq", equal)
    REGISTER_PYBIND11_MPS_BINARY_OP("ne", notEqual)
    REGISTER_PYBIND11_MPS_BINARY_OP("ge", greaterThanOrEqualTo)
    REGISTER_PYBIND11_MPS_BINARY_OP("gt", greaterThan)
    REGISTER_PYBIND11_MPS_BINARY_OP("le", lessThanOrEqualTo)
    REGISTER_PYBIND11_MPS_BINARY_OP("lt", lessThan)

    // Bitwise Ops
    REGISTER_PYBIND11_MPS_BITWISE_BINARY_OP("bitwise_and", AND)
    REGISTER_PYBIND11_MPS_BITWISE_BINARY_OP("bitwise_or", OR)
    REGISTER_PYBIND11_MPS_BITWISE_BINARY_OP("bitwise_xor", XOR)

    .def("bitwise_not", [](MPSGraphModule& self, PyMPSGraphTensor* input) {
      return self.bitwiseNotTensor(static_cast<MPSGraphTensor*>(input), "bitwise_not");
    })

    // Boolean Binary Ops
    REGISTER_PYBIND11_MPS_BINARY_OP("eq", equal)
    REGISTER_PYBIND11_MPS_BINARY_OP("ne", notEqual)
    REGISTER_PYBIND11_MPS_BINARY_OP("le", lessThanOrEqualTo)
    REGISTER_PYBIND11_MPS_BINARY_OP("lt", lessThan)
    REGISTER_PYBIND11_MPS_BINARY_OP("ge", greaterThanOrEqualTo)
    REGISTER_PYBIND11_MPS_BINARY_OP("gt", greaterThan)

    // Unary Ops

    REGISTER_PYBIND11_MPS_UNARY_OP("abs", absolute)
    REGISTER_PYBIND11_MPS_UNARY_OP("exp", exponent)
    REGISTER_PYBIND11_MPS_UNARY_OP("exp2", exponentBase2)
    REGISTER_PYBIND11_MPS_UNARY_OP("reciprocal", reciprocal)
    REGISTER_PYBIND11_MPS_UNARY_OP("sqrt", squareRoot)
    REGISTER_PYBIND11_MPS_UNARY_OP("neg", negative)
    REGISTER_PYBIND11_MPS_UNARY_OP("log", logarithm)
    REGISTER_PYBIND11_MPS_UNARY_OP("log10", logarithmBase10)
    REGISTER_PYBIND11_MPS_UNARY_OP("log2", logarithmBase2)
    REGISTER_PYBIND11_MPS_UNARY_OP("erf", erf)
    REGISTER_PYBIND11_MPS_UNARY_OP("floor", floor)
    REGISTER_PYBIND11_MPS_UNARY_OP("ceil", ceil)
    REGISTER_PYBIND11_MPS_UNARY_OP("rsqrt", reverseSquareRoot)
    REGISTER_PYBIND11_MPS_UNARY_OP("sin", sin)
    REGISTER_PYBIND11_MPS_UNARY_OP("sign", sign)
    REGISTER_PYBIND11_MPS_UNARY_OP("sigmoid", sigmoid)
    REGISTER_PYBIND11_MPS_UNARY_OP("cos", cos)
    REGISTER_PYBIND11_MPS_UNARY_OP("tan", tan)
    REGISTER_PYBIND11_MPS_UNARY_OP("asin", asin)
    REGISTER_PYBIND11_MPS_UNARY_OP("acos", acos)
    REGISTER_PYBIND11_MPS_UNARY_OP("atan", atan)
    REGISTER_PYBIND11_MPS_UNARY_OP("sinh", sinh)
    REGISTER_PYBIND11_MPS_UNARY_OP("cosh", cosh)
    REGISTER_PYBIND11_MPS_UNARY_OP("tanh", tanh)
    REGISTER_PYBIND11_MPS_UNARY_OP("asinh", asinh)
    REGISTER_PYBIND11_MPS_UNARY_OP("acosh", acosh)
    REGISTER_PYBIND11_MPS_UNARY_OP("atanh", atanh)
    REGISTER_PYBIND11_MPS_UNARY_OP("isinf", isInfinite)
    REGISTER_PYBIND11_MPS_UNARY_OP("isnan", isNaN)
    REGISTER_PYBIND11_MPS_UNARY_OP("round", round)

    .def("div", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor) {
      return self.div_mode_template(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), c10::nullopt, "div");
    })
    .def("fmod", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor) {
      return self.div_mode_template(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), "trunc", "fmod_mps_out");
    })
    .def("floor_divide", [](MPSGraphModule& self, PyMPSGraphTensor* primaryTensor, PyMPSGraphTensor* secondaryTensor) {
      return self.div_mode_template(static_cast<MPSGraphTensor*>(primaryTensor), static_cast<MPSGraphTensor*>(secondaryTensor), "floor", "floor_divide");
    })

    //
    // Graph debug methods.
    //
    .def("printGraph", &MPSGraphModule::printGraph)

    //
    // Serialization / deserialization methods.
    //
    .def("serialize", &MPSGraphModule::serialize);

  // Export `MPSDataType` Objective-C enum to python.
  py::enum_<MPSDataType>(m, "MPSDataType")
    .value("MPSDataTypeTypeInvalid", MPSDataType::MPSDataTypeFloatBit)
    .value("MPSDataTypeFloat32",     MPSDataType::MPSDataTypeFloat32)
    .value("MPSDataTypeFloat16",     MPSDataType::MPSDataTypeFloat16)
    .value("MPSDataTypeInt32",       MPSDataType::MPSDataTypeInt32)
    .value("MPSDataTypeInt64",       MPSDataType::MPSDataTypeInt64)
    .value("MPSDataTypeInt16",       MPSDataType::MPSDataTypeInt16)
    .value("MPSDataTypeInt8",        MPSDataType::MPSDataTypeInt8)
    .value("MPSDataTypeUInt8",       MPSDataType::MPSDataTypeUInt8)
    .value("MPSDataTypeBool",        MPSDataType::MPSDataTypeBool)
    .export_values();
}

} // namespace
} // namespace mps
