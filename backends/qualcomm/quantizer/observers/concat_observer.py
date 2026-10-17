# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from executorch.backends.qualcomm.utils.constants import DEFAULT_EPS_FP32
from torchao.quantization.pt2e import (
    FakeQuantize,
    FakeQuantizeBase,
    UniformQuantizationObserverBase,
)
from torchao.quantization.pt2e.observer import PartialWrapper


class ConcatObserver(UniformQuantizationObserverBase):
    """
    Fetch maximum data range of all tensors to be concatenated
    """

    def __init__(
        self,
        node_name,
        graph,
        dtype=torch.uint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=DEFAULT_EPS_FP32,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        # get concat node and its inputs
        self.concat_node = [node for node in graph.nodes if node.name == node_name][0]
        self.input_nodes = self.concat_node.args[0]
        self.input_observers = []

    def __deepcopy__(self, memo):
        # input_nodes still point at the original owning_module,
        # so a lazy resolve on the copy would bind to the original's observers.
        if len(self.input_observers) == 0:
            self._collect_input_observers()

        # Share by reference: deep-copying fx.Nodes recurses the whole graph.
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = (
                value
                if key in ("concat_node", "input_nodes")
                else copy.deepcopy(value, memo)
            )
        return new

    def _collect_input_observers(self):
        # we cannot do this in constructor since observers have not appeared
        for node in self.input_nodes:
            obs_node = list(
                filter(lambda user: user != self.concat_node, node.users.keys())
            )[0]
            self.input_observers.append(
                getattr(obs_node.graph.owning_module, obs_node.name)
            )

    def forward(self, x_orig):
        # calculate the min / max first
        min_val, max_val = torch.aminmax(x_orig.detach())
        self.min_val = min(self.min_val, min_val)
        self.max_val = max(self.max_val, max_val)

        if len(self.input_observers) == 0:
            # collect observers first if they are not cached
            self._collect_input_observers()

        # push min/max to each input's observer. Under QAT it's wrapped in a
        # FakeQuantize, whose range lives on activation_post_process.
        for observer in self.input_observers:
            target = (
                observer.activation_post_process
                if isinstance(observer, FakeQuantizeBase)
                else observer
            )
            # a fixed-qparams input has no min_val/max_val
            if not hasattr(target, "min_val") or not hasattr(target, "max_val"):
                continue
            target.min_val.copy_(self.min_val)
            target.max_val.copy_(self.max_val)

        return x_orig

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)


class ConcatFakeQuantize(FakeQuantize):
    """FakeQuantize using ConcatObserver, so concat's output is fake-quantized
    under QAT instead of just carrying a (never simulated) observer."""

    def __init__(self, node_name, graph, **kwargs):
        super().__init__(
            observer=ConcatObserver, node_name=node_name, graph=graph, **kwargs
        )


def concat_observer_ctr(node_name, graph, output_activation):
    """
    PTQ keeps using ConcatObserver; QAT swaps to ConcatFakeQuantize so the
    output stays fake-quantized, but both share the same max-range behavior.
    """
    kwargs = {"node_name": node_name, "graph": graph}
    # with_args() can be chained, so unwrap until we hit the class.
    unwrapped = output_activation.observer_or_fake_quant_ctr
    while isinstance(unwrapped, PartialWrapper):
        unwrapped = unwrapped.p.func
    if isinstance(unwrapped, type) and issubclass(unwrapped, FakeQuantizeBase):
        return ConcatFakeQuantize.with_args(**kwargs)
    return ConcatObserver.with_args(**kwargs)
