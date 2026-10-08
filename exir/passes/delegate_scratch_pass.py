# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import Counter
from typing import List, Optional, Union

import torch
from executorch.exir.backend.backend_details import (
    DelegateScratchSpec,
    DelegateStateSpec,
)
from executorch.exir.delegate import executorch_call_delegate, is_lowered_module
from executorch.exir.memory import (
    DELEGATE_SCRATCH_META_KEY,
    DELEGATE_STATE_META_KEY,
    PlannedScratch,
    PlannedState,
)
from executorch.exir.memory_planning import get_node_tensor_specs
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from torch.fx.passes.infra.pass_base import PassBase, PassResult


def _lowered_module(
    graph_module: torch.fx.GraphModule, delegate_call: torch.fx.Node
) -> Optional[torch.nn.Module]:
    """The LoweredBackendModule a delegate call names in its first argument."""
    if not delegate_call.args:
        return None
    lowered_node = delegate_call.args[0]
    if not isinstance(lowered_node, torch.fx.Node) or lowered_node.op != "get_attr":
        return None
    module = getattr(graph_module, lowered_node.target, None)
    return module if is_lowered_module(module) else None


def _delegate_calls(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is executorch_call_delegate
    ]


class DelegateScratchSpecPass(PassBase):
    """Materializes the scratch and state buffers a backend asked for in ``preprocess``.

    Each ``DelegateScratchSpec`` and ``DelegateStateSpec`` becomes a
    one-dimensional ``uint8`` ``TensorSpec`` on the delegate call's metadata,
    which is the only thing the memory planner knows how to place. The emitter
    then serializes the planned location onto the ``DelegateCall`` rather than
    as a value, so the backend receives byte ranges rather than tensors.
    Scratch and state go under separate keys, because the planner gives them
    different lifetimes and the runtime delivers them in separate lists.

    The specs are metadata rather than graph nodes, so the delegate call's
    arguments still match the signature its lowered module was partitioned
    with and a later pass is free to retrace.

    Runs after ``SpecPropPass``, and after ``PropagateDevicePass`` for a
    delegate that is not on the CPU, since that is where the device a scratch
    buffer inherits comes from. Both are checked rather than declared:
    ``to_executorch()`` orders its passes as a list, not as a constraint graph.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if isinstance(module, torch.fx.GraphModule):
                modified |= self._annotate_graph(module, module is graph_module)
        return PassResult(graph_module, modified)

    def _annotate_graph(
        self, graph_module: torch.fx.GraphModule, is_root: bool
    ) -> bool:
        modified = False
        delegate_calls = _delegate_calls(graph_module)
        call_sites = Counter(
            _lowered_module(graph_module, node) for node in delegate_calls
        )
        for node in delegate_calls:
            # Cleared before anything can skip the node: to_executorch() writes
            # its lowered graph back into the edge program, so a second call
            # finds the first call's specs still here, including when the
            # backend has since stopped asking for any.
            modified |= node.meta.pop(DELEGATE_SCRATCH_META_KEY, None) is not None
            modified |= node.meta.pop(DELEGATE_STATE_META_KEY, None) is not None

            lowered_module = _lowered_module(graph_module, node)
            if lowered_module is None:
                continue
            if lowered_module.scratch_specs:
                node.meta[DELEGATE_SCRATCH_META_KEY] = [
                    PlannedScratch(label=declared.label, spec=spec)
                    for declared, spec in zip(
                        lowered_module.scratch_specs,
                        self._make_specs(node, lowered_module.scratch_specs),
                    )
                ]
                modified = True
            if lowered_module.state_specs:
                self._check_state_supported(
                    node, lowered_module, is_root, call_sites[lowered_module]
                )
                node.meta[DELEGATE_STATE_META_KEY] = [
                    PlannedState(
                        label=declared.label,
                        initial_data=declared.initial_data,
                        spec=spec,
                    )
                    for declared, spec in zip(
                        lowered_module.state_specs,
                        self._make_specs(node, lowered_module.state_specs),
                    )
                ]
                modified = True
        return modified

    def _check_state_supported(
        self,
        node: torch.fx.Node,
        lowered_module: torch.nn.Module,
        is_root: bool,
        call_sites: int,
    ) -> None:
        backend = lowered_module.backend_id
        if not is_root:
            # A submodule is planned with its own node numbering, so a lifetime
            # spanning its graph would not span the method's.
            raise RuntimeError(
                f"Backend {backend} declares state buffers on delegate call "
                f"{node.name} inside a control-flow submodule, which is not supported."
            )
        if call_sites > 1:
            # Each call site gets its own state, where the eager model would
            # share one buffer between them.
            raise RuntimeError(
                f"Backend {backend} declares state buffers but its lowered module "
                f"is called from {call_sites} places, so each would see separate "
                "state. Lower each stateful region once."
            )
        for spec in get_node_tensor_specs(node):
            if spec.device != DeviceType.CPU:
                raise RuntimeError(
                    f"Backend {backend} declares state buffers on delegate call "
                    f"{node.name}, which runs on {spec.device.name}. State buffers "
                    "are only supported on the CPU."
                )

    def _make_specs(
        self,
        node: torch.fx.Node,
        declared: List[Union[DelegateScratchSpec, DelegateStateSpec]],
    ) -> List[TensorSpec]:
        delegate_specs = get_node_tensor_specs(node)
        if not delegate_specs:
            raise RuntimeError(
                f"Delegate call {node.name} has no tensor specs. "
                "DelegateScratchSpecPass must run after SpecPropPass, and after "
                "PropagateDevicePass for a delegate that is not on the CPU."
            )
        specs = []
        for buffer in declared:
            spec = TensorSpec(dtype=torch.uint8, shape=torch.Size([buffer.nbytes]))
            spec.device = delegate_specs[0].device
            spec.device_index = delegate_specs[0].device_index
            specs.append(spec)
        return specs
