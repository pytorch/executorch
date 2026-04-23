#pragma once

namespace executorch::backends::xnnpack::graph {

enum class Operator {
    // Binary elementwise
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    CopySign,
    SquaredDifference,
    PReLU,
    Modulus,
    Atan2,
    Pow,

    // Unary elementwise
    Abs,
    Negate,
    Clamp,
    Ceiling,
    Floor,
    Round,
    Square,
    SquareRoot,
    ReciprocalSquareRoot,
    Exp,
    Log,
    Sigmoid,
    Tanh,
    ELU,
    GELU,
    HardSwish,
    LeakyReLU,
    Sine,
    Cosine,
    Sign,
    ReLU,

    // Linear algebra
    Linear,
    BatchMatrixMultiply,

    // Convolution
    Conv2d,
    ConvTranspose2d,

    // Pooling
    AvgPool2d,
    AdaptiveAvgPool2d,
    MaxPool2d,

    // Reduction
    Softmax,
    Mean,
    Sum,

    // Shape / memory
    Reshape,
    View,
    Transpose,
    Permute,
    Slice,
    Cat,
    Chunk,
    Unsqueeze,
    Expand,
    Clone,
    Pad,

    // Quantization
    Quantize,
    Dequantize,

    // In-tree
    LayerNorm,
};

}
