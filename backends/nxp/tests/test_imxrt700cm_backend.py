# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    AddTensorConverter,
    SoftmaxConverter,
)
from executorch.backends.nxp.imxrt700cm.imxrt700cm_pipeline import lower_to_imxrt700cm
from executorch.backends.nxp.tests.executors import OverrideTargetSupportCheck
from executorch.examples.nxp.experimental.cifar_net.cifar_net import CifarNet
from executorch.examples.nxp.models.mobilenet_v2 import MobilenetV2
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

CortexMAdd = exir_ops.edge.cortex_m.quantized_add.default
CortexMDequantize = exir_ops.edge.cortex_m.dequantize_per_tensor.default
CortexMMaximum = exir_ops.edge.cortex_m.maximum.default
CortexMMinimum = exir_ops.edge.cortex_m.minimum.default
CortexMQuantize = exir_ops.edge.cortex_m.quantize_per_tensor.default
CortexMSoftmax = exir_ops.edge.cortex_m.softmax.default
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate


def test_cifar_net__float_io():
    model_base = CifarNet()
    model = model_base.get_eager_model().eval().to(memory_format=torch.channels_last)
    input_shape = (1, 3, 32, 32)
    target = "imxrt700"

    # The SoftMax in CifarNet is currently unsupported by Neutron, so it is handled by CortexM. But we prohibit
    #  Neutron from delegating it anyway, to make sure this test will be passing even with future SoftMax enablement.
    def _unsupported(*_) -> bool:
        return False

    with OverrideTargetSupportCheck(
        SoftmaxConverter, new_target_support_check=_unsupported
    ):
        epm = lower_to_imxrt700cm(model, input_shape, target=target)

    nodes = list(epm.exported_program().graph.nodes)
    assert nodes[1].target == CortexMQuantize
    assert nodes[3].target == ExecutorchDelegateCall
    assert nodes[4].target == operator.getitem
    assert nodes[5].target == CortexMSoftmax
    assert nodes[6].target == CortexMDequantize


def test_mobile_net__simulated_unsupported_add():
    model_base = MobilenetV2(use_random_dataset=True)
    model = model_base.get_eager_model().eval().to(memory_format=torch.channels_last)
    input_shape = (1, 3, 224, 224)
    target = "imxrt700"

    def _new_add_target_support(node: Node, *_) -> bool:
        # Do not delegate this one specific node to simulate an unsupported case.
        return node.meta["torch_fn"][0] != "add_2"

    with OverrideTargetSupportCheck(
        AddTensorConverter, new_target_support_check=_new_add_target_support
    ):
        epm = lower_to_imxrt700cm(model, input_shape, target=target)

    nodes = list(epm.exported_program().graph.nodes)
    assert nodes[1].target == CortexMQuantize
    assert nodes[3].target == ExecutorchDelegateCall
    assert nodes[4].target == operator.getitem
    assert nodes[5].target == operator.getitem
    assert nodes[6].target == CortexMAdd and nodes[6].meta["torch_fn"][0] == "add_2"
    assert nodes[8].target == ExecutorchDelegateCall
    assert nodes[9].target == operator.getitem
    assert nodes[10].target == CortexMDequantize


class AddMinimumMaximumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x + x
        y = torch.minimum(x, y)
        x = torch.maximum(x, y)
        return x


def test_add_minimum_maximum():
    # The `aten.minimum` and `aten.maximum` are not supported by Neutron, but they are supported by Cortex-M.
    # The 2 consecutive nodes are used to test that the `IMXRT700Quantizer` correctly uses the `CortexMQuantizer` to
    #  quantize nodes which were not quantized by the `NeutronQuantizer`, and were not marked with
    #  `NXP_NEUTRON_BACKEND_IGNORE`.
    model = AddMinimumMaximumModel().eval()
    input_shape = (24,)
    target = "imxrt700"

    epm = lower_to_imxrt700cm(model, input_shape, target=target)

    nodes = list(epm.exported_program().graph.nodes)
    assert nodes[1].target == CortexMQuantize
    assert nodes[3].target == ExecutorchDelegateCall
    assert nodes[4].target == operator.getitem
    assert nodes[5].target == CortexMMinimum
    assert nodes[6].target == CortexMMaximum
    assert nodes[7].target == CortexMDequantize


class MulMinimumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x * x
        x = torch.minimum(x, y)
        return x


def test_mul_minimum():
    # The `aten.minimum` is not supported by Neutron, but it is supported by Cortex-M.
    # The `Mul` is supported by Neutron, but it requires a specific quantization parameters, which are not compatible
    #  with the CortexM Minimum node after. Therefore, extra Dequantize -> Quantize is inserted.
    # TODO Look into whether this can be optimized.(EIEX-806)
    model = MulMinimumModel().eval()
    input_shape = (24,)
    target = "imxrt700"

    epm = lower_to_imxrt700cm(model, input_shape, target=target)

    nodes = list(epm.exported_program().graph.nodes)
    print(nodes)
    assert nodes[1].target == CortexMQuantize
    assert nodes[3].target == ExecutorchDelegateCall
    assert nodes[4].target == operator.getitem
    assert nodes[5].target == CortexMDequantize
    assert nodes[6].target == CortexMQuantize
    assert nodes[7].target == CortexMMinimum
    assert nodes[8].target == CortexMDequantize
