# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import types
from contextlib import contextmanager

import torch
import torchao
from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
    PerBlockParamObserver,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torchao.quantization.pt2e import PerChannelMinMaxObserver


class SeqMseModule(torch.nn.Module):
    """
    Args:
        nominal_weight: Tensor
            nominal parameters from operator
        nominal_bias: Tensor
            nominal parameters from operator
        operator: fx.Node
            operator to be executed
        observer: UniformQuantizationObserverBase
            parameter observer (specific for weight)
        num_candidates: int
            grids to search minimal mse loss
    """

    def __init__(
        self,
        nominal_weight,
        nominal_bias,
        operator,
        observer,
        num_candidates,
    ):
        super().__init__()
        self.nominal_weight = nominal_weight
        self.nominal_bias = nominal_bias
        self.observer = observer
        self.steps = torch.linspace(
            1 / num_candidates, 1, steps=num_candidates
        ).tolist()
        self.operator = self._make_operator(operator)
        self.best_candidate_step = 1.0

    def _make_operator(self, aten_op):
        if aten_op.target == torch.ops.aten.conv2d.default:
            stride = [1, 1] if len(aten_op.args) < 4 else aten_op.args[3]
            padding = [0, 0] if len(aten_op.args) < 5 else aten_op.args[4]
            dilation = [1, 1] if len(aten_op.args) < 6 else aten_op.args[5]
            groups = 1 if len(aten_op.args) < 7 else aten_op.args[6]
            has_bias = self.nominal_bias is not None
            module = torch.nn.Conv2d(
                in_channels=self.nominal_weight.shape[1]
                * groups,  # equivalent to input_tensor.shape[1]
                out_channels=self.nominal_weight.shape[0],
                kernel_size=self.nominal_weight.shape[-2:],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=has_bias,
            )
            module.weight.data = self.nominal_weight
            if has_bias:
                module.bias.data = self.nominal_bias
            return module
        else:
            raise NotImplementedError(f"target of {aten_op.target} is not implemented")

    def _per_block_qdq(self, scale, zero_point):
        return torchao.quantization.quant_primitives._fake_quantize_affine(
            input=self.nominal_weight,
            block_size=self.observer.block_size,
            scale=scale,
            zero_point=zero_point,
            quant_dtype=self.observer.dtype,
            quant_min=self.observer.quant_min,
            quant_max=self.observer.quant_max,
        )

    def _per_channel_qdq(self, scale, zero_point):
        return torch.fake_quantize_per_channel_affine(
            input=self.nominal_weight,
            scale=scale,
            zero_point=zero_point,
            axis=0,
            quant_min=self.observer.quant_min,
            quant_max=self.observer.quant_max,
        )

    def _fake_quant(self, scale, zero_point):
        dispatcher = {
            PerChannelMinMaxObserver: self._per_channel_qdq,
            PerBlockParamObserver: self._per_block_qdq,
        }
        return dispatcher[type(self.observer)](scale, zero_point)

    def _find_best_candidate(self, nominal_input, nominal_output):
        # calculate current baseline
        scale, zero_point = self.observer.calculate_qparams()
        zero_point = zero_point.to(torch.int32)
        self.operator.weight.data = self._fake_quant(scale, zero_point)
        candidate, current_loss = (
            1,
            torch.nn.functional.mse_loss(
                self.operator(nominal_input), nominal_output
            ).item(),
        )
        for step in self.steps:
            self.operator.weight.data = self._fake_quant(scale * step, zero_point)
            loss = torch.nn.functional.mse_loss(
                self.operator(nominal_input), nominal_output
            ).item()
            if loss < current_loss:
                candidate, current_loss = step, loss
        return candidate

    def forward(self, nominal_input, nominal_output):
        self.best_candidate_step = self._find_best_candidate(
            nominal_input=nominal_input, nominal_output=nominal_output
        )


class InsertSeqMse(ExportPass):
    """
    Insert Seq Mse Observer to find the best quant config for certain node's weight.
    """

    seq_mse_ops = {torch.ops.aten.conv2d.default}

    def __init__(self, num_candidates=1000):
        super(InsertSeqMse, self).__init__()
        self.num_candidates = num_candidates

    def _insert_seq_mse(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        count = 0
        for node in graph_module.graph.nodes:
            if node.target in self.seq_mse_ops:
                # extract observer
                weight_node_obs = node.args[1]
                observer = getattr(graph_module, weight_node_obs.name)
                # extract parameters
                weight_node = weight_node_obs.args[0]
                weight_tensor = graph_module.get_parameter(weight_node.target).detach()
                bias_tensor = None
                if len(node.args) > 2 and node.args[2] is not None:
                    bias_tensor = graph_module.get_parameter(
                        node.args[2].args[0].target
                    ).detach()

                with graph_module.graph.inserting_after(node):
                    seq_mse_mod = SeqMseModule(
                        nominal_weight=weight_tensor,
                        nominal_bias=bias_tensor,
                        operator=node,
                        observer=observer,
                        num_candidates=self.num_candidates,
                    )
                    module_name = f"seq_mse_{count}"
                    count += 1
                    setattr(graph_module, module_name, seq_mse_mod)
                    input_nodes = (node.args[0], node)
                    graph_module.graph.create_node(
                        "call_module", module_name, input_nodes, {}
                    )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert_seq_mse(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)


class RemoveSeqMse(ExportPass):
    """
    Remove Seq Mse before invoking convert_pt2e and update final quantization encoding.
    """

    def __init__(self):
        super(RemoveSeqMse, self).__init__()

    def _remove_seq_mse(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        node_to_erase = []
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                # try extracting SeqMse module
                module = getattr(graph_module, node.target)
                if isinstance(module, SeqMseModule):
                    # rewrite observer method for pre-calculated scale
                    scale, zero_point = module.observer.calculate_qparams()
                    module.observer.updated_encoding = (
                        scale * module.best_candidate_step,
                        zero_point,
                    )
                    module.observer.calculate_qparams = types.MethodType(
                        lambda s: s.updated_encoding, module.observer
                    )
                    node_to_erase.append(node)

        for node in node_to_erase:
            graph_module.graph.erase_node(node)

    def call(self, graph_module: torch.fx.GraphModule):
        self._remove_seq_mse(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)


@contextmanager
def SeqMSE(prepared_gm, num_candidates):
    prepared_gm = InsertSeqMse(num_candidates)(prepared_gm).graph_module
    try:
        yield
    finally:
        prepared_gm = RemoveSeqMse()(prepared_gm).graph_module
