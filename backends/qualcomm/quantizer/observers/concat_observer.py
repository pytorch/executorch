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
        # Share the live-graph fx.Nodes: copying them walks the whole node list
        # (RecursionError on large graphs) and its FakeTensor metadata.
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = (
                value
                if key in ("concat_node", "input_nodes")
                else copy.deepcopy(value, memo)
            )
        return new

    def forward(self, x_orig):
        # calculate the min / max first
        min_val, max_val = torch.aminmax(x_orig.detach())
        self.min_val = min(self.min_val, min_val)
        self.max_val = max(self.max_val, max_val)

        if len(self.input_observers) == 0:
            # collect observers first if they are not cached
            # we cannot do this in constructor since observers have not appeared
            for node in self.input_nodes:
                obs_node = list(
                    filter(lambda user: user != self.concat_node, node.users.keys())
                )[0]
                self.input_observers.append(
                    getattr(obs_node.graph.owning_module, obs_node.name)
                )

        # update min / max for all observers of input nodes. Under QAT the input
        # is a FakeQuantize wrapping the real observer, so the range it reads
        # back in calculate_qparams() lives on activation_post_process, not on
        # the FakeQuantize module itself.
        for observers in self.input_observers:
            target = (
                observers.activation_post_process
                if isinstance(observers, FakeQuantizeBase)
                else observers
            )
            target.min_val = self.min_val
            target.max_val = self.max_val

        return x_orig

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)


class ConcatFakeQuantize(FakeQuantize):
    """FakeQuantize paired with ConcatObserver, so a cat's output is still
    fake-quantized under QAT instead of carrying a bare (never simulated)
    observer.
    """

    def __init__(self, node_name, graph, **kwargs):
        super().__init__(
            observer=ConcatObserver, node_name=node_name, graph=graph, **kwargs
        )


def concat_observer_ctr(node_name, graph, output_activation):
    """Build the observer/fake-quant ctr for a cat's output qspec.

    Under PTQ ``output_activation``'s ctr is a plain observer, so the cat
    output stays a ``ConcatObserver`` as before. Under QAT it is a
    ``FakeQuantize``, so swap in ``ConcatFakeQuantize`` instead, which keeps
    the output fake-quantized like every other activation while giving it the
    same max-range-of-all-inputs behavior ConcatObserver gives PTQ.
    """
    kwargs = {"node_name": node_name, "graph": graph}
    base = getattr(output_activation.observer_or_fake_quant_ctr, "p", None)
    if base is not None and issubclass(base.func, FakeQuantizeBase):
        return ConcatFakeQuantize.with_args(**kwargs)
    return ConcatObserver.with_args(**kwargs)
