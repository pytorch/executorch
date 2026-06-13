#pragma once

namespace executorch::backends::xnnpack::graph {

/*
 * Operators known to the CPU backend.
 */
enum class Operator {
  // Binary elementwise
  Add = 0,
  Subtract = 1,
  Multiply = 2,
  Divide = 3,
  Maximum = 4,
  Minimum = 5,
  CopySign = 6,
  SquaredDifference = 7,
  PReLU = 8,
  Modulus = 9,
  Atan2 = 10,
  Pow = 11,

  // Unary elementwise
  Abs = 12,
  Negate = 13,
  Clamp = 14,
  Ceiling = 15,
  Floor = 16,
  Round = 17,
  Square = 18,
  SquareRoot = 19,
  ReciprocalSquareRoot = 20,
  Exp = 21,
  Log = 22,
  Sigmoid = 23,
  Tanh = 24,
  ELU = 25,
  GELU = 26,
  HardSwish = 27,
  LeakyReLU = 28,
  Sine = 29,
  Cosine = 30,
  Sign = 31,
  ReLU = 32,

  // Linear
  Linear = 33,
  BatchMatrixMultiply = 34,

  // Convolution
  Conv2d = 35,
  ConvTranspose2d = 36,

  // Pooling
  AvgPool2d = 37,
  AdaptiveAvgPool2d = 38,
  MaxPool2d = 39,

  // Reduction
  Softmax = 40,
  Mean = 41,
  Sum = 42,

  // Shape / memory
  Reshape = 43,
  View = 44,
  Transpose = 45,
  Permute = 46,
  Slice = 47,
  Cat = 48,
  Chunk = 49,
  Unsqueeze = 50,
  Expand = 51,
  Clone = 52,
  Pad = 53,

  // Quantization
  Quantize = 54,
  Dequantize = 55,

  // Norms
  LayerNorm = 56,

  DepthwiseConv2d = 57,
  StaticResizeBilinear2D = 58,
};

} // namespace executorch::backends::xnnpack::graph
