# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict
from pprint import pformat

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.cortex_m.test.tester import McuTestCase, ramp_tensor


_QUANT_CONFIG_INT8 = get_symmetric_quantization_config()
_QUANT_CONFIG_INT16 = get_symmetric_a16w8_quantization_config()
_INT8_QSPEC = _QUANT_CONFIG_INT8.output_activation
_PER_TENSOR_TARGETS = {
    "quantized_decomposed.quantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_tensor.default",
}
_PER_CHANNEL_TARGET = "quantized_decomposed.dequantize_per_channel.default"


class SubOp(torch.nn.Module):
    def forward(self, x, y):
        return x - y


def _get_quantizer() -> TOSAQuantizer:
    """
    Returns a TOSAQuantizer configured for int8 quantization with SubOp unquantized.
    """
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(_QUANT_CONFIG_INT8)
    quantizer.set_module_type(SubOp, None)
    return quantizer


def _collect_quant_params(nodes):
    found = defaultdict(Counter)
    for node in nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target in _PER_TENSOR_TARGETS:
            scale, zero_point, qmin, qmax, dtype = node.args[1:6]
            params = (
                round(float(scale), 9),
                int(zero_point),
                int(qmin),
                int(qmax),
                dtype,
            )
            found[target][params] += 1
        elif target == _PER_CHANNEL_TARGET:
            axis = int(node.args[3])
            qmin = int(node.args[4])
            qmax = int(node.args[5])
            dtype = node.args[6]
            found[target][(axis, qmin, qmax, dtype)] += 1
    return {target: dict(counter) for target, counter in found.items()}


def _check_quant_params(pipeline, expected):
    if expected is None:
        return
    nodes = pipeline.tester.get_artifact().module().graph.nodes
    found = _collect_quant_params(nodes)
    if found != expected:
        raise AssertionError(
            f"Quant params mismatch.\nExpected: {pformat(expected)}\nFound: {pformat(found)}"
        )


class SharedQspecMulipleClusters(torch.nn.Module):
    """Three linear shared qspec clusters."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 8},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 8},
        "aten.add.Tensor": {_INT8_QSPEC: 2},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 2,
            (0.031357229, 0, -128, 127, torch.int8): 4,
            (0.062714458, 0, -128, 127, torch.int8): 2,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 2,
            (0.031357229, 0, -128, 127, torch.int8): 4,
            (0.062714458, 0, -128, 127, torch.int8): 2,
        },
    }

    def forward(self, x):
        x = torch.clone(x)
        x = x + x
        x = torch.clone(x)
        x = torch.clone(x)
        x = torch.clone(x)
        x = x + x
        x = torch.transpose(x, 2, 1)
        return x


class SharedQspecInputForkNonShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with both inputs as non-shared qspecs."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 4},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 4},
    }
    inputs_qspecs = {None: 2}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 4,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 4,
        },
    }

    def forward(self, x, y):
        z = torch.maximum(x, y)
        return torch.flatten(z)


class SharedQspecInputForkShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with both inputs as shared qspecs."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 5},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 5},
    }
    inputs_qspecs = {None: 2}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 5,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 5,
        },
    }

    def forward(self, x, y):
        x = torch.clone(x)
        y = torch.permute(y, (0, 1, 3, 2))
        z = torch.minimum(x, y)
        return z


class SharedQspecInputForkXShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as shared qspec."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 4},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 4},
    }
    inputs_qspecs = {None: 2}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 4,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 4,
        },
    }

    def forward(self, x, y):
        x = torch.t_copy(x)
        z = torch.maximum(x, y)
        return z


class SharedQspecInputForkYShared(torch.nn.Module):
    """Shared qspec cluster with an input fork with right input as shared qspec."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 5},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 5},
    }
    inputs_qspecs = {None: 2}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 5,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.01959827, -26, -128, 127, torch.int8): 5,
        },
    }

    def forward(self, x, y):
        y = torch.clone(y)
        z = torch.minimum(x, y)
        return torch.squeeze(z)


class SharedQspecInputForkXConstant(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as global constant."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 2},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 3},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.027437577, -55, -128, 127, torch.int8): 3,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.027437577, -55, -128, 127, torch.int8): 2,
        },
    }
    constant = torch.tensor(5.0)

    def forward(self, x):
        return torch.minimum(self.constant, x)


class SharedQspecInputForkYConstant(torch.nn.Module):
    """Shared qspec cluster with an input fork with left input as local constant."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 2},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 3},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.027437577, -55, -128, 127, torch.int8): 3,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.027437577, -55, -128, 127, torch.int8): 2,
        },
    }

    def forward(self, x):
        return torch.maximum(x, torch.tensor(5.0))


class SharedQspecOutputForkNonShared(torch.nn.Module):
    """Shared qspec cluster with an output fork with both outputs as non-shared qspecs."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 3},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 4},
        "aten.add.Tensor": {_INT8_QSPEC: 1},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 3,
            (0.031357229, 0, -128, 127, torch.int8): 1,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 2,
            (0.031357229, 0, -128, 127, torch.int8): 1,
        },
    }

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        y = x + x
        return x, y


class SharedQspecOutputForkShared(torch.nn.Module):
    """Shared qspec cluster with an output fork with both outputs as shared qspecs."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 4},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 6},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 6,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 4,
        },
    }

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        y = torch.clone(x)
        z = torch.permute_copy(x, (0, 2, 1, 3))
        return y, z, x


class SharedQspecManyForks(torch.nn.Module):
    """Shared qspec cluster with a number of forks to test more complex structures."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 6},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 9},
        "aten.t.default": {None: 1},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.086232387, 104, -128, 127, torch.int8): 9,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.086232387, 104, -128, 127, torch.int8): 6,
        },
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.maximum(x, x1)
        x3 = torch.maximum(x, torch.t(x2))
        x4 = torch.minimum(x2, x3)
        return x4


class SharedQspecSurroundedQuantizedOp(torch.nn.Module):
    """An annotated int8 surrounded by a shared qspec cluster forcing input/output qparams to be equal."""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 4},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 5},
        "aten.add.Tensor": {_INT8_QSPEC: 1},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (1.019109964, 123, -128, 127, torch.int8): 5,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (1.019109964, 123, -128, 127, torch.int8): 4,
        },
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.add(x1, x1)
        x3 = torch.maximum(x1, x2)
        return x3


class SharedQspecSurroundedQuantizedOpConstant(torch.nn.Module):
    """ """

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 5},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 6},
        "aten.ones.default": {_INT8_QSPEC: 1},
        "aten.add.Tensor": {_INT8_QSPEC: 1},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.003921569, -128, -128, 127, torch.int8): 1,
            (0.01959827, -26, -128, 127, torch.int8): 5,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.003921569, -128, -128, 127, torch.int8): 1,
            (0.01959827, -26, -128, 127, torch.int8): 4,
        },
    }

    def forward(self, x):
        x1 = torch.clone(x)
        x2 = torch.add(x1, torch.ones(2, 2))
        x3 = torch.maximum(x1, x2)
        return x3


class SharedQspecSub(torch.nn.Module):
    """A shared qspec node with float input"""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 2},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 2},
        "aten.sub.Tensor": {None: 1},
    }
    inputs_qspecs = {None: 2}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.035276882, -128, -128, 127, torch.int8): 2,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.035276882, -128, -128, 127, torch.int8): 2,
        },
    }

    def __init__(self):
        super().__init__()
        self.sub = SubOp()

    def forward(self, x, y):
        return torch.clone(self.sub(x, y))


class SharedQspecCompetingQspecs(torch.nn.Module):
    """A shared qspec node with per-channel/per-tensor annotated nodes as inputs"""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 3},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 4},
        "aten.conv2d.default": {_INT8_QSPEC: 1},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_channel.default": {
            (0, -2147483647, 2147483647, torch.int32): 1,
            (0, -127, 127, torch.int8): 1,
        },
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.046643879, -87, -128, 127, torch.int8): 4,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.046643879, -87, -128, 127, torch.int8): 3,
        },
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv.weight.data = torch.tensor(
            [
                [[[0.1]], [[-0.2]], [[0.3]]],
                [[[-0.1]], [[0.2]], [[-0.3]]],
                [[[0.05]], [[-0.05]], [[0.15]]],
            ],
            dtype=self.conv.weight.dtype,
        )
        self.conv.bias.data = torch.tensor([0.0, 0.1, -0.1], dtype=self.conv.bias.dtype)

    def forward(self, x):
        return torch.cat([self.conv(x), x], dim=1)


class SharedQspecNoQspecs(torch.nn.Module):
    """A shared qspec node with float input/outputs"""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 2},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 2},
        "aten.sub.Tensor": {None: 2},
    }
    inputs_qspecs = {None: 1}
    outputs_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.000244141, -128, -128, 127, torch.int8): 2,
        },
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.000244141, -128, -128, 127, torch.int8): 2,
        },
    }

    def __init__(self):
        super().__init__()
        self.sub = SubOp()

    def forward(self, x):
        z = torch.clone(self.sub(x, x))
        return self.sub(z, z)


class Int8Branch(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class Int16Branch(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class MixedMaximumInt8Int16(torch.nn.Module):
    """A shared qspec node with int16/int8 inputs"""

    qspecs = {
        "quantized_decomposed.quantize_per_tensor.default": {None: 6},
        "quantized_decomposed.dequantize_per_tensor.default": {None: 6},
    }
    input_qspecs = {None: 1}
    output_qspecs = {None: 1}
    quant_params = {
        "quantized_decomposed.quantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 4,
            (0.000244141, 0, -32767, 32767, torch.int16): 2,
        },
        "quantized_decomposed.dequantize_per_tensor.default": {
            (0.015678614, 0, -128, 127, torch.int8): 4,
            (0.000244141, 0, -32767, 32767, torch.int16): 2,
        },
    }

    def __init__(self):
        super().__init__()
        self.int16 = Int16Branch()
        self.int8 = Int8Branch()

    def forward(self, x):
        return torch.maximum(self.int16(x), self.int8(x))


test_cases = {
    "multiple_clusters": McuTestCase(
        SharedQspecMulipleClusters(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "input_fork_non_shared": McuTestCase(
        SharedQspecInputForkNonShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)), ramp_tensor(-1, 3, (2, 3, 4))),
    ),
    "input_fork_shared": McuTestCase(
        SharedQspecInputForkShared(),
        (ramp_tensor(-2, 2, (2, 3, 4, 5)), ramp_tensor(-1, 3, (2, 3, 5, 4))),
    ),
    "input_fork_x_shared": McuTestCase(
        SharedQspecInputForkXShared(),
        (ramp_tensor(-2, 2, (3, 4)), ramp_tensor(-1, 3, (4, 3))),
    ),
    "input_fork_y_shared": McuTestCase(
        SharedQspecInputForkYShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)), ramp_tensor(-1, 3, (2, 3, 4))),
    ),
    "input_fork_x_constant": McuTestCase(
        SharedQspecInputForkXConstant(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "input_fork_y_constant": McuTestCase(
        SharedQspecInputForkYConstant(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "surrounded_quantized_op": McuTestCase(
        SharedQspecSurroundedQuantizedOp(),
        (ramp_tensor(-128, 2, (2, 3, 4)),),
    ),
    "surrounded_quantized_op_constant": McuTestCase(
        SharedQspecSurroundedQuantizedOpConstant(),
        (ramp_tensor(-2, 2, (2, 2)),),
    ),
    "output_fork_non_shared": McuTestCase(
        SharedQspecOutputForkNonShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "output_fork_shared": McuTestCase(
        SharedQspecOutputForkShared(),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "many_forks": McuTestCase(
        SharedQspecManyForks(),
        (ramp_tensor(-20, 2, (4, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_shared_qspec_quantizer(test_case):
    """
    Test that ops which does not change dynamic range are able to use int8 portable kernels.
    """
    pipeline = QuantizationPipeline(
        test_case.model,
        test_case.example_inputs,
        quantizer=_get_quantizer(),
        qspecs=test_case.model.qspecs,
        input_qspecs=test_case.model.inputs_qspecs,
        output_qspecs=test_case.model.outputs_qspecs,
    )
    pipeline.run()
    _check_quant_params(pipeline, test_case.model.quant_params)


float_test_cases = {
    "non-quantized_op": McuTestCase(
        SharedQspecSub(),
        (ramp_tensor(0, 10, (5, 5)), ramp_tensor(0, 1, (5, 5))),
    ),
    "competing_qspecs": McuTestCase(
        SharedQspecCompetingQspecs(),
        (ramp_tensor(0, 10, (1, 3, 5, 5)).to(memory_format=torch.channels_last),),
    ),
    "no_qspecs": McuTestCase(
        SharedQspecNoQspecs(),
        (ramp_tensor(0, 10, (1, 3, 5, 5)),),
    ),
}


@parametrize("test_case", float_test_cases)
def test_shared_qspec_quantizer_no_qspecs(test_case):
    """
    Test that ops which does not change dynamic range are able to use int8 portable kernels.
    """
    pipeline = QuantizationPipeline(
        test_case.model,
        test_case.example_inputs,
        quantizer=_get_quantizer(),
        qspecs=test_case.model.qspecs,
        input_qspecs=test_case.model.inputs_qspecs,
        output_qspecs=test_case.model.outputs_qspecs,
    )
    pipeline.run()
    _check_quant_params(pipeline, test_case.model.quant_params)


def test_maximum_mixed_int8_int16_inputs():
    model = MixedMaximumInt8Int16()
    inputs = (ramp_tensor(-2, 2, (2, 3, 4)),)

    quantizer = _get_quantizer()
    quantizer.set_module_type(Int16Branch, _QUANT_CONFIG_INT16)
    quantizer.set_module_type(Int8Branch, _QUANT_CONFIG_INT8)

    pipeline = QuantizationPipeline(
        model,
        inputs,
        quantizer=quantizer,
        qspecs=model.qspecs,
        input_qspecs=model.input_qspecs,
        output_qspecs=model.output_qspecs,
    )
    pipeline.run()
    _check_quant_params(pipeline, model.quant_params)
