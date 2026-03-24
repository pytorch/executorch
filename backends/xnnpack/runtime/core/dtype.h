#pragma once

namespace executorch::backends::xnnpack::core {

enum class DType {
    Float32,

    // Quantized — signed symmetric
    QInt8Sym,
    QInt4Sym,
    QInt32Sym,

    // Quantized — unsigned asymmetric
    QUInt8Asym,
};

}
