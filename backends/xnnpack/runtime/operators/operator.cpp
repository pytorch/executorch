#include <executorch/backends/xnnpack/runtime/operators/operator.h>

#include <executorch/backends/xnnpack/runtime/operators/layer_norm.h>

#include <cstdlib>

namespace executorch::backends::xnnpack::operators {

std::unique_ptr<Operator> create_operator(graph::Operator op) {
    switch (op) {
        case graph::Operator::LayerNorm:
            return std::make_unique<LayerNorm>();
        default:
            abort();
    }
}

}
