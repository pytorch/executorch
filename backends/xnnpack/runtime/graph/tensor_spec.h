#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace executorch::backends::xnnpack::graph {

struct DimSizeSpec {
    struct Term {
        SymIntHandle sym;
        int64_t coefficient;

        bool operator==(const Term& o) const {
            return sym == o.sym && coefficient == o.coefficient;
        }
    };

    std::vector<Term> coeffs;
    int64_t offset = 0;

    static DimSizeSpec constant(int64_t value) {
        return { {}, value };
    }

    static DimSizeSpec sym(SymIntHandle s) {
        return { {{ s, 1 }}, 0 };
    }

    bool is_constant() const { return coeffs.empty(); }

    bool operator==(const DimSizeSpec& o) const {
        return coeffs == o.coeffs && offset == o.offset;
    }
};

struct TensorSpec {
    core::DType dtype;
    std::vector<DimSizeSpec> sizes;
    std::optional<core::QuantParams> quant_params;

    bool operator==(const TensorSpec& o) const {
        return dtype == o.dtype && sizes == o.sizes && quant_params == o.quant_params;
    }
};

}
